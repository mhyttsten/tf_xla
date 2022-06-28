#ifndef CXX11_SRC_FIXEDPOINT_PACKETMATHAVX_H_
#define CXX11_SRC_FIXEDPOINT_PACKETMATHAVX_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh() {
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

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<__m256i, 10> Packet32q8i;
typedef eigen_packet_wrapper<__m128i, 11> Packet16q8i;

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
    HasMin = 0,
    HasMax = 0,
    HasConj = 0,
    HasSetLinear = 0
  };
};

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
EIGEN_STRONG_INLINE Packet32q8i pset1<Packet32q8i>(const QInt8& from) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_0(mht_0_v, 235, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pset1<Packet32q8i>");

  return _mm256_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q8i ploadu<Packet32q8i>(const QInt8* from) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_1(mht_1_v, 242, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "ploadu<Packet32q8i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i ploadu<Packet16q8i>(const QInt8* from) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_2(mht_2_v, 250, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "ploadu<Packet16q8i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(from));
}

template <>
EIGEN_STRONG_INLINE Packet32q8i pload<Packet32q8i>(const QInt8* from) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_3(mht_3_v, 259, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pload<Packet32q8i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i pload<Packet16q8i>(const QInt8* from) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_4(mht_4_v, 267, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pload<Packet16q8i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(
      reinterpret_cast<const __m128i*>(from));
}

template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet32q8i& from) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_5(mht_5_v, 276, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pstoreu<QInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet16q8i& from) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_6(mht_6_v, 284, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pstoreu<QInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.m_val);
}

template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet32q8i& from) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_7(mht_7_v, 293, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pstore<QInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet16q8i& from) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVXDTh mht_8(mht_8_v, 301, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX.h", "pstore<QInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.m_val);
}

typedef __m256 Packet8f;

template <>
struct type_casting_traits<float, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8i
pcast<Packet8f, Packet32q8i>(const Packet8f& a, const Packet8f& b,
                             const Packet8f& c, const Packet8f& d) {
  const __m256i a_conv = _mm256_cvtps_epi32(a);
  const __m256i b_conv = _mm256_cvtps_epi32(b);
  const __m256i c_conv = _mm256_cvtps_epi32(c);
  const __m256i d_conv = _mm256_cvtps_epi32(d);
  __m128i low = _mm256_castsi256_si128(a_conv);
  __m128i high = _mm256_extractf128_si256(a_conv, 1);
  __m128i tmp = _mm_packs_epi32(low, high);
  __m128i low2 = _mm256_castsi256_si128(b_conv);
  __m128i high2 = _mm256_extractf128_si256(b_conv, 1);
  __m128i tmp2 = _mm_packs_epi32(low2, high2);
  __m128i converted_low = _mm_packs_epi16(tmp, tmp2);
  low = _mm256_castsi256_si128(c_conv);
  high = _mm256_extractf128_si256(c_conv, 1);
  tmp = _mm_packs_epi32(low, high);
  low2 = _mm256_castsi256_si128(d_conv);
  high2 = _mm256_extractf128_si256(d_conv, 1);
  tmp2 = _mm_packs_epi32(low2, high2);
  __m128i converted_high = _mm_packs_epi16(tmp, tmp2);
  return _mm256_insertf128_si256(_mm256_castsi128_si256(converted_low),
                                 converted_high, 1);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_PACKETMATHAVX_H_
