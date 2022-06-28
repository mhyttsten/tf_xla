#ifndef CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
#define CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh() {
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

#ifdef _MSC_VER

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#endif

inline int _mm256_extract_epi16_N0(const __m256i X) {
  return _mm_extract_epi16(_mm256_extractf128_si256(X, 0 >> 3), 0 % 8);
}

inline int _mm256_extract_epi16_N1(const __m256i X) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_0(mht_0_v, 184, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "_mm256_extract_epi16_N1");

  return _mm_extract_epi16(_mm256_extractf128_si256(X, 1 >> 3), 1 % 8);
}

inline int _mm256_extract_epi8_N0(const __m256i X) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_1(mht_1_v, 191, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "_mm256_extract_epi8_N0");

  return _mm_extract_epi8(_mm256_extractf128_si256((X), 0 >> 4), 0 % 16);
}

inline int _mm256_extract_epi8_N1(const __m256i X) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_2(mht_2_v, 198, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "_mm256_extract_epi8_N1");

  return _mm_extract_epi8(_mm256_extractf128_si256((X), 1 >> 4), 1 % 16);
}

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<__m256i, 20> Packet32q8i;
typedef eigen_packet_wrapper<__m256i, 21> Packet16q16i;
typedef eigen_packet_wrapper<__m256i, 22> Packet32q8u;
typedef eigen_packet_wrapper<__m128i, 23> Packet16q8i;
typedef eigen_packet_wrapper<__m128i, 25> Packet16q8u;
typedef eigen_packet_wrapper<__m128i, 26> Packet8q16i;
typedef eigen_packet_wrapper<__m256i, 27> Packet8q32i;
typedef eigen_packet_wrapper<__m128i, 28> Packet4q32i;

#ifndef EIGEN_VECTORIZE_AVX512
template <>
struct packet_traits<QInt8> : default_packet_traits {
  typedef Packet32q8i type;
  typedef Packet16q8i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QUInt8> : default_packet_traits {
  typedef Packet32q8u type;
  typedef Packet16q8u half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QInt16> : default_packet_traits {
  typedef Packet16q16i type;
  typedef Packet8q16i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QInt32> : default_packet_traits {
  typedef Packet8q32i type;
  typedef Packet4q32i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,
  };
  enum {
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
#endif

template <>
struct unpacket_traits<Packet32q8i> {
  typedef QInt8 type;
  typedef Packet16q8i half;
  enum {
    size = 32,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet16q8i> {
  typedef QInt8 type;
  typedef Packet16q8i half;
  enum {
    size = 16,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet16q16i> {
  typedef QInt16 type;
  typedef Packet8q16i half;
  enum {
    size = 16,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet8q16i> {
  typedef QInt16 type;
  typedef Packet8q16i half;
  enum {
    size = 8,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet32q8u> {
  typedef QUInt8 type;
  typedef Packet16q8u half;
  enum {
    size = 32,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet8q32i> {
  typedef QInt32 type;
  typedef Packet4q32i half;
  enum {
    size = 8,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

// Unaligned load
template <>
EIGEN_STRONG_INLINE Packet32q8i ploadu<Packet32q8i>(const QInt8* from) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_3(mht_3_v, 383, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet32q8i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i ploadu<Packet16q8i>(const QInt8* from) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_4(mht_4_v, 391, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet16q8i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q8u ploadu<Packet32q8u>(const QUInt8* from) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_5(mht_5_v, 399, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet32q8u>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q16i ploadu<Packet16q16i>(const QInt16* from) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_6(mht_6_v, 407, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet16q16i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q16i ploadu<Packet8q16i>(const QInt16* from) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_7(mht_7_v, 415, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet8q16i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i ploadu<Packet8q32i>(const QInt32* from) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_8(mht_8_v, 423, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "ploadu<Packet8q32i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}

// Aligned load
template <>
EIGEN_STRONG_INLINE Packet32q8i pload<Packet32q8i>(const QInt8* from) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_9(mht_9_v, 433, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet32q8i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i pload<Packet16q8i>(const QInt8* from) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_10(mht_10_v, 441, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet16q8i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q8u pload<Packet32q8u>(const QUInt8* from) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_11(mht_11_v, 449, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet32q8u>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pload<Packet16q16i>(const QInt16* from) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_12(mht_12_v, 457, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet16q16i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q16i pload<Packet8q16i>(const QInt16* from) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_13(mht_13_v, 465, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet8q16i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pload<Packet8q32i>(const QInt32* from) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_14(mht_14_v, 473, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pload<Packet8q32i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}

// Unaligned store
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet32q8i& from) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_15(mht_15_v, 483, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet16q8i& from) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_16(mht_16_v, 491, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QUInt8>(QUInt8* to, const Packet32q8u& from) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_17(mht_17_v, 499, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QUInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet16q16i& from) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_18(mht_18_v, 507, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QInt16>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet8q16i& from) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_19(mht_19_v, 515, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QInt16>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt32>(QInt32* to, const Packet8q32i& from) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_20(mht_20_v, 523, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstoreu<QInt32>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}

// Aligned store
template <>
EIGEN_STRONG_INLINE void pstore<QInt32>(QInt32* to, const Packet8q32i& from) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_21(mht_21_v, 533, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QInt32>");

  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet16q16i& from) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_22(mht_22_v, 541, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QInt16>");

  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet8q16i& from) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_23(mht_23_v, 549, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QInt16>");

  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QUInt8>(QUInt8* to, const Packet32q8u& from) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_24(mht_24_v, 557, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QUInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet32q8i& from) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_25(mht_25_v, 565, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet16q8i& from) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_26(mht_26_v, 573, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pstore<QInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.m_val);
}

// Extract first element.
template <>
EIGEN_STRONG_INLINE QInt32 pfirst<Packet8q32i>(const Packet8q32i& a) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_27(mht_27_v, 583, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pfirst<Packet8q32i>");

  return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
template <>
EIGEN_STRONG_INLINE QInt16 pfirst<Packet16q16i>(const Packet16q16i& a) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_28(mht_28_v, 590, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pfirst<Packet16q16i>");

  return _mm256_extract_epi16_N0(a.m_val);
}
template <>
EIGEN_STRONG_INLINE QUInt8 pfirst<Packet32q8u>(const Packet32q8u& a) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_29(mht_29_v, 597, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pfirst<Packet32q8u>");

  return static_cast<uint8_t>(_mm256_extract_epi8_N0(a.m_val));
}
template <>
EIGEN_STRONG_INLINE QInt8 pfirst<Packet32q8i>(const Packet32q8i& a) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_30(mht_30_v, 604, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pfirst<Packet32q8i>");

  return _mm256_extract_epi8_N0(a.m_val);
}

// Initialize to constant value.
template <>
EIGEN_STRONG_INLINE Packet32q8i pset1<Packet32q8i>(const QInt8& from) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_31(mht_31_v, 613, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pset1<Packet32q8i>");

  return _mm256_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q8u pset1<Packet32q8u>(const QUInt8& from) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_32(mht_32_v, 620, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pset1<Packet32q8u>");

  return _mm256_set1_epi8(static_cast<uint8_t>(from.value));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pset1<Packet8q32i>(const QInt32& from) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_33(mht_33_v, 627, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pset1<Packet8q32i>");

  return _mm256_set1_epi32(from.value);
}

// Basic arithmetic packet ops for QInt32.
template <>
EIGEN_STRONG_INLINE Packet8q32i padd<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_34(mht_34_v, 637, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "padd<Packet8q32i>");

  return _mm256_add_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pset1<Packet16q16i>(const QInt16& from) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_35(mht_35_v, 644, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pset1<Packet16q16i>");

  return _mm256_set1_epi16(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i psub<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_36(mht_36_v, 652, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "psub<Packet8q32i>");

  return _mm256_sub_epi32(a.m_val, b.m_val);
}
// Note: mullo truncates the result to 32 bits.
template <>
EIGEN_STRONG_INLINE Packet8q32i pmul<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_37(mht_37_v, 661, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmul<Packet8q32i>");

  return _mm256_mullo_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pnegate<Packet8q32i>(const Packet8q32i& a) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_38(mht_38_v, 668, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pnegate<Packet8q32i>");

  return _mm256_sub_epi32(_mm256_setzero_si256(), a.m_val);
}

// Min and max.
template <>
EIGEN_STRONG_INLINE Packet8q32i pmin<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_39(mht_39_v, 678, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmin<Packet8q32i>");

  return _mm256_min_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pmax<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_40(mht_40_v, 686, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmax<Packet8q32i>");

  return _mm256_max_epi32(a.m_val, b.m_val);
}

template <>
EIGEN_STRONG_INLINE Packet16q16i pmin<Packet16q16i>(const Packet16q16i& a,
                                                    const Packet16q16i& b) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_41(mht_41_v, 695, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmin<Packet16q16i>");

  return _mm256_min_epi16(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pmax<Packet16q16i>(const Packet16q16i& a,
                                                    const Packet16q16i& b) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_42(mht_42_v, 703, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmax<Packet16q16i>");

  return _mm256_max_epi16(a.m_val, b.m_val);
}

template <>
EIGEN_STRONG_INLINE Packet32q8u pmin<Packet32q8u>(const Packet32q8u& a,
                                                  const Packet32q8u& b) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_43(mht_43_v, 712, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmin<Packet32q8u>");

  return _mm256_min_epu8(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet32q8u pmax<Packet32q8u>(const Packet32q8u& a,
                                                  const Packet32q8u& b) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_44(mht_44_v, 720, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmax<Packet32q8u>");

  return _mm256_max_epu8(a.m_val, b.m_val);
}

template <>
EIGEN_STRONG_INLINE Packet32q8i pmin<Packet32q8i>(const Packet32q8i& a,
                                                  const Packet32q8i& b) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_45(mht_45_v, 729, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmin<Packet32q8i>");

  return _mm256_min_epi8(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet32q8i pmax<Packet32q8i>(const Packet32q8i& a,
                                                  const Packet32q8i& b) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_46(mht_46_v, 737, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "pmax<Packet32q8i>");

  return _mm256_max_epi8(a.m_val, b.m_val);
}

// Reductions.
template <>
EIGEN_STRONG_INLINE QInt32 predux_min<Packet8q32i>(const Packet8q32i& a) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_47(mht_47_v, 746, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_min<Packet8q32i>");

  __m256i tmp = _mm256_min_epi32(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi32(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return pfirst<Packet8q32i>(
      _mm256_min_epi32(tmp, _mm256_shuffle_epi32(tmp, 1)));
}
template <>
EIGEN_STRONG_INLINE QInt32 predux_max<Packet8q32i>(const Packet8q32i& a) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_48(mht_48_v, 757, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_max<Packet8q32i>");

  __m256i tmp = _mm256_max_epi32(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi32(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return pfirst<Packet8q32i>(
      _mm256_max_epi32(tmp, _mm256_shuffle_epi32(tmp, 1)));
}

template <>
EIGEN_STRONG_INLINE QInt16 predux_min<Packet16q16i>(const Packet16q16i& a) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_49(mht_49_v, 769, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_min<Packet16q16i>");

  __m256i tmp = _mm256_min_epi16(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi16(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epi16(tmp, _mm256_shuffle_epi32(tmp, 1));
  return std::min(_mm256_extract_epi16_N0(tmp), _mm256_extract_epi16_N1(tmp));
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_max<Packet16q16i>(const Packet16q16i& a) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_50(mht_50_v, 780, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_max<Packet16q16i>");

  __m256i tmp = _mm256_max_epi16(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi16(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epi16(tmp, _mm256_shuffle_epi32(tmp, 1));
  return std::max(_mm256_extract_epi16_N0(tmp), _mm256_extract_epi16_N1(tmp));
}

template <>
EIGEN_STRONG_INLINE QUInt8 predux_min<Packet32q8u>(const Packet32q8u& a) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_51(mht_51_v, 792, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_min<Packet32q8u>");

  __m256i tmp = _mm256_min_epu8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epu8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epu8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_min_epu8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::min(static_cast<uint8_t>(_mm256_extract_epi8_N0(tmp)),
                  static_cast<uint8_t>(_mm256_extract_epi8_N1(tmp)));
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_max<Packet32q8u>(const Packet32q8u& a) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_52(mht_52_v, 806, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_max<Packet32q8u>");

  __m256i tmp = _mm256_max_epu8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epu8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epu8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_max_epu8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::max(static_cast<uint8_t>(_mm256_extract_epi8_N0(tmp)),
                  static_cast<uint8_t>(_mm256_extract_epi8_N1(tmp)));
}

template <>
EIGEN_STRONG_INLINE QInt8 predux_min<Packet32q8i>(const Packet32q8i& a) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_53(mht_53_v, 821, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_min<Packet32q8i>");

  __m256i tmp = _mm256_min_epi8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epi8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_min_epi8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::min(_mm256_extract_epi8_N0(tmp), _mm256_extract_epi8_N1(tmp));
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_max<Packet32q8i>(const Packet32q8i& a) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_54(mht_54_v, 834, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "predux_max<Packet32q8i>");

  __m256i tmp = _mm256_max_epi8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epi8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_max_epi8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::max(_mm256_extract_epi8_N0(tmp), _mm256_extract_epi8_N1(tmp));
}

// Vectorized scaling of Packet32q8i by float.
template <>
struct scalar_product_op<QInt32, double> : binary_op_base<QInt32, double> {
  typedef typename ScalarBinaryOpTraits<QInt32, double>::ReturnType result_type;
#ifndef EIGEN_SCALAR_BINARY_OP_PLUGIN
  EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
#else
  scalar_product_op() { EIGEN_SCALAR_BINARY_OP_PLUGIN }
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type
  operator()(const QInt32& a, const double& b) const {
    return a * b;
  }

  EIGEN_STRONG_INLINE const Packet8q32i packetOp(const Packet8q32i& a,
                                                 const double& b) const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX2DTh mht_55(mht_55_v, 862, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX2.h", "packetOp");

    __m256d scale = _mm256_set1_pd(b);
    __m256d a_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(a));
    __m128i result_lo = _mm256_cvtpd_epi32(_mm256_mul_pd(scale, a_lo));
    __m256d a_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(a, 1));
    __m128i result_hi = _mm256_cvtpd_epi32(_mm256_mul_pd(scale, a_hi));
    return _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi,
                                   1);
  }
};

template <>
struct functor_traits<scalar_product_op<QInt32, double>> {
  enum { Cost = 4 * NumTraits<float>::MulCost, PacketAccess = true };
};

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
