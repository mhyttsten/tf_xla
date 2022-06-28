#ifndef CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_
#define CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh() {
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


#include "PacketMathAVX2.h"

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<__m512i, 30> Packet64q8i;
typedef eigen_packet_wrapper<__m512i, 31> Packet32q16i;
typedef eigen_packet_wrapper<__m512i, 32> Packet64q8u;
typedef eigen_packet_wrapper<__m512i, 33> Packet16q32i;

template <>
struct packet_traits<QInt8> : default_packet_traits {
  typedef Packet64q8i type;
  typedef Packet32q8i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 64,
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
  typedef Packet64q8u type;
  typedef Packet32q8u half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 64,
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
  typedef Packet32q16i type;
  typedef Packet16q16i half;
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
struct packet_traits<QInt32> : default_packet_traits {
  typedef Packet16q32i type;
  typedef Packet8q32i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
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

template <>
struct unpacket_traits<Packet64q8i> {
  typedef QInt8 type;
  typedef Packet32q8i half;
  enum {
    size = 64,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet32q16i> {
  typedef QInt16 type;
  typedef Packet16q16i half;
  enum {
    size = 32,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet64q8u> {
  typedef QUInt8 type;
  typedef Packet32q8u half;
  enum {
    size = 64,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet16q32i> {
  typedef QInt32 type;
  typedef Packet8q32i half;
  enum {
    size = 16,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};

// Unaligned load
template <>
EIGEN_STRONG_INLINE Packet64q8i ploadu<Packet64q8i>(const QInt8* from) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_0(mht_0_v, 319, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "ploadu<Packet64q8i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q16i ploadu<Packet32q16i>(const QInt16* from) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_1(mht_1_v, 327, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "ploadu<Packet32q16i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet64q8u ploadu<Packet64q8u>(const QUInt8* from) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_2(mht_2_v, 335, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "ploadu<Packet64q8u>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i ploadu<Packet16q32i>(const QInt32* from) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_3(mht_3_v, 343, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "ploadu<Packet16q32i>");

  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}

// Aligned load
template <>
EIGEN_STRONG_INLINE Packet64q8i pload<Packet64q8i>(const QInt8* from) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_4(mht_4_v, 353, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pload<Packet64q8i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pload<Packet32q16i>(const QInt16* from) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_5(mht_5_v, 361, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pload<Packet32q16i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pload<Packet64q8u>(const QUInt8* from) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_6(mht_6_v, 369, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pload<Packet64q8u>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pload<Packet16q32i>(const QInt32* from) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_7(mht_7_v, 377, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pload<Packet16q32i>");

  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}

// Unaligned store
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet64q8i& from) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_8(mht_8_v, 387, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstoreu<QInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet32q16i& from) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_9(mht_9_v, 395, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstoreu<QInt16>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QUInt8>(QUInt8* to, const Packet64q8u& from) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_10(mht_10_v, 403, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstoreu<QUInt8>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt32>(QInt32* to, const Packet16q32i& from) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_11(mht_11_v, 411, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstoreu<QInt32>");

  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}

// Aligned store
template <>
EIGEN_STRONG_INLINE void pstore<QInt32>(QInt32* to, const Packet16q32i& from) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_12(mht_12_v, 421, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstore<QInt32>");

  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QUInt8>(QUInt8* to, const Packet64q8u& from) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_13(mht_13_v, 429, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstore<QUInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet64q8i& from) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_14(mht_14_v, 437, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstore<QInt8>");

  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet32q16i& from) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_15(mht_15_v, 445, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pstore<QInt16>");

  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}

// Extract first element.
template <>
EIGEN_STRONG_INLINE QInt32 pfirst<Packet16q32i>(const Packet16q32i& a) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_16(mht_16_v, 455, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pfirst<Packet16q32i>");

  return _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(a, 0));
}
template <>
EIGEN_STRONG_INLINE QUInt8 pfirst<Packet64q8u>(const Packet64q8u& a) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_17(mht_17_v, 462, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pfirst<Packet64q8u>");

  return static_cast<uint8_t>(
      _mm_extract_epi8(_mm512_extracti32x4_epi32(a.m_val, 0), 0));
}
template <>
EIGEN_STRONG_INLINE QInt8 pfirst<Packet64q8i>(const Packet64q8i& a) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_18(mht_18_v, 470, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pfirst<Packet64q8i>");

  return _mm_extract_epi8(_mm512_extracti32x4_epi32(a.m_val, 0), 0);
}
template <>
EIGEN_STRONG_INLINE QInt16 pfirst<Packet32q16i>(const Packet32q16i& a) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_19(mht_19_v, 477, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pfirst<Packet32q16i>");

  return _mm_extract_epi16(_mm512_extracti32x4_epi32(a.m_val, 0), 0);
}

// Initialize to constant value.
template <>
EIGEN_STRONG_INLINE Packet64q8i pset1<Packet64q8i>(const QInt8& from) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_20(mht_20_v, 486, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pset1<Packet64q8i>");

  return _mm512_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pset1<Packet32q16i>(const QInt16& from) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_21(mht_21_v, 493, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pset1<Packet32q16i>");

  return _mm512_set1_epi16(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pset1<Packet64q8u>(const QUInt8& from) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_22(mht_22_v, 500, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pset1<Packet64q8u>");

  return _mm512_set1_epi8(static_cast<uint8_t>(from.value));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pset1<Packet16q32i>(const QInt32& from) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_23(mht_23_v, 507, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pset1<Packet16q32i>");

  return _mm512_set1_epi32(from.value);
}

// Basic arithmetic packet ops for QInt32.
template <>
EIGEN_STRONG_INLINE Packet16q32i padd<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_24(mht_24_v, 517, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "padd<Packet16q32i>");

  return _mm512_add_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i psub<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_25(mht_25_v, 525, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "psub<Packet16q32i>");

  return _mm512_sub_epi32(a.m_val, b.m_val);
}
// Note: mullo truncates the result to 32 bits.
template <>
EIGEN_STRONG_INLINE Packet16q32i pmul<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_26(mht_26_v, 534, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmul<Packet16q32i>");

  return _mm512_mullo_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pnegate<Packet16q32i>(const Packet16q32i& a) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_27(mht_27_v, 541, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pnegate<Packet16q32i>");

  return _mm512_sub_epi32(_mm512_setzero_si512(), a.m_val);
}

// Min and max.
template <>
EIGEN_STRONG_INLINE Packet16q32i pmin<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_28(mht_28_v, 551, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmin<Packet16q32i>");

  return _mm512_min_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pmax<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_29(mht_29_v, 559, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmax<Packet16q32i>");

  return _mm512_max_epi32(a.m_val, b.m_val);
}

template <>
EIGEN_STRONG_INLINE Packet64q8u pmin<Packet64q8u>(const Packet64q8u& a,
                                                  const Packet64q8u& b) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_30(mht_30_v, 568, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmin<Packet64q8u>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epu8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epu8(ap0, bp0);
  __m256i r1 = _mm256_min_epu8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pmax<Packet64q8u>(const Packet64q8u& a,
                                                  const Packet64q8u& b) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_31(mht_31_v, 586, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmax<Packet64q8u>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epu8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epu8(ap0, bp0);
  __m256i r1 = _mm256_max_epu8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet64q8i pmin<Packet64q8i>(const Packet64q8i& a,
                                                  const Packet64q8i& b) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_32(mht_32_v, 605, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmin<Packet64q8i>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epi8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epi8(ap0, bp0);
  __m256i r1 = _mm256_min_epi8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pmin<Packet32q16i>(const Packet32q16i& a,
                                                    const Packet32q16i& b) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_33(mht_33_v, 623, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmin<Packet32q16i>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epi16(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epi16(ap0, bp0);
  __m256i r1 = _mm256_min_epi16(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet64q8i pmax<Packet64q8i>(const Packet64q8i& a,
                                                  const Packet64q8i& b) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_34(mht_34_v, 641, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmax<Packet64q8i>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epi8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epi8(ap0, bp0);
  __m256i r1 = _mm256_max_epi8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pmax<Packet32q16i>(const Packet32q16i& a,
                                                    const Packet32q16i& b) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_35(mht_35_v, 659, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "pmax<Packet32q16i>");

#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epi16(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epi16(ap0, bp0);
  __m256i r1 = _mm256_max_epi16(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}

// Reductions.
template <>
EIGEN_STRONG_INLINE QInt32 predux_min<Packet16q32i>(const Packet16q32i& a) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_36(mht_36_v, 678, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_min<Packet16q32i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi32(_mm_min_epi32(lane0, lane1), _mm_min_epi32(lane2, lane3));
  res = _mm_min_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  return pfirst(res);
}
template <>
EIGEN_STRONG_INLINE QInt32 predux_max<Packet16q32i>(const Packet16q32i& a) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_37(mht_37_v, 693, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_max<Packet16q32i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi32(_mm_max_epi32(lane0, lane1), _mm_max_epi32(lane2, lane3));
  res = _mm_max_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  return pfirst(res);
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_min<Packet32q16i>(const Packet32q16i& a) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_38(mht_38_v, 708, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_min<Packet32q16i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi16(_mm_min_epi16(lane0, lane1), _mm_min_epi16(lane2, lane3));
  res = _mm_min_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int16_t>(w >> 16), static_cast<std::int16_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_max<Packet32q16i>(const Packet32q16i& a) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_39(mht_39_v, 725, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_max<Packet32q16i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi16(_mm_max_epi16(lane0, lane1), _mm_max_epi16(lane2, lane3));
  res = _mm_max_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::max(
      {static_cast<std::int16_t>(w >> 16), static_cast<std::int16_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_min<Packet64q8u>(const Packet64q8u& a) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_40(mht_40_v, 742, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_min<Packet64q8u>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epu8(_mm_min_epu8(lane0, lane1), _mm_min_epu8(lane2, lane3));
  res = _mm_min_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::uint8_t>(w >> 24), static_cast<std::uint8_t>(w >> 16),
       static_cast<std::uint8_t>(w >> 8), static_cast<std::uint8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_max<Packet64q8u>(const Packet64q8u& a) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_41(mht_41_v, 760, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_max<Packet64q8u>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epu8(_mm_max_epu8(lane0, lane1), _mm_max_epu8(lane2, lane3));
  res = _mm_max_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::max(
      {static_cast<std::uint8_t>(w >> 24), static_cast<std::uint8_t>(w >> 16),
       static_cast<std::uint8_t>(w >> 8), static_cast<std::uint8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_min<Packet64q8i>(const Packet64q8i& a) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_42(mht_42_v, 778, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_min<Packet64q8i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi8(_mm_min_epi8(lane0, lane1), _mm_min_epi8(lane2, lane3));
  res = _mm_min_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int8_t>(w >> 24), static_cast<std::int8_t>(w >> 16),
       static_cast<std::int8_t>(w >> 8), static_cast<std::int8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_max<Packet64q8i>(const Packet64q8i& a) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSPacketMathAVX512DTh mht_43(mht_43_v, 796, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/PacketMathAVX512.h", "predux_max<Packet64q8i>");

  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi8(_mm_max_epi8(lane0, lane1), _mm_max_epi8(lane2, lane3));
  res = _mm_max_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int8_t>(w >> 24), static_cast<std::int8_t>(w >> 16),
       static_cast<std::int8_t>(w >> 8), static_cast<std::int8_t>(w)});
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_
