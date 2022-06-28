#ifndef CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_
#define CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh() {
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


namespace Eigen {
namespace internal {

typedef __m512 Packet16f;
typedef __m512i Packet16i;

template <>
struct type_casting_traits<QInt32, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16q32i>(const Packet16q32i& a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh mht_0(mht_0_v, 185, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX512.h", "pcast<Packet16q32i>");

  return _mm512_cvtepi32_ps(a.m_val);
}

template <>
struct type_casting_traits<float, QInt32> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet16q32i pcast<Packet16f>(const Packet16f& a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh mht_1(mht_1_v, 198, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX512.h", "pcast<Packet16f>");

  return _mm512_cvtps_epi32(a);
}

template <>
struct type_casting_traits<float, QInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q16i pcast<Packet16f>(const Packet16f& a,
                                                  const Packet16f& b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh mht_2(mht_2_v, 212, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX512.h", "pcast<Packet16f>");

  Packet16i a_int = _mm512_cvtps_epi32(a);
  Packet16i b_int = _mm512_cvtps_epi32(b);
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_packs_epi32(a_int, b_int);
#else
  Packet8i ab_int16_low = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_castsi512_si256(a_int),
                         _mm512_castsi512_si256(b_int)),
      _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i ab_int16_high = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_extracti32x8_epi32(a_int, 1),
                         _mm512_extracti32x8_epi32(b_int, 1)),
      _MM_SHUFFLE(0, 2, 1, 3));
  return _mm512_inserti32x8(_mm512_castsi256_si512(ab_int16_low), ab_int16_high,
                            1);
#endif
}

template <>
struct type_casting_traits<float, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8i pcast<Packet16f>(const Packet16f& a,
                                                 const Packet16f& b,
                                                 const Packet16f& c,
                                                 const Packet16f& d) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX512DTh mht_3(mht_3_v, 243, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX512.h", "pcast<Packet16f>");

  Packet16i a_int = _mm512_cvtps_epi32(a);
  Packet16i b_int = _mm512_cvtps_epi32(b);
  Packet16i c_int = _mm512_cvtps_epi32(c);
  Packet16i d_int = _mm512_cvtps_epi32(d);
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_packs_epi16(_mm512_packs_epi32(a_int, b_int),
                            _mm512_packs_epi32(c_int, d_int));
#else
  Packet8i ab_int16_low = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_castsi512_si256(a_int),
                         _mm512_castsi512_si256(b_int)),
      _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i cd_int16_low = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_castsi512_si256(c_int),
                         _mm512_castsi512_si256(d_int)),
      _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i ab_int16_high = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_extracti32x8_epi32(a_int, 1),
                         _mm512_extracti32x8_epi32(b_int, 1)),
      _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i cd_int16_high = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(_mm512_extracti32x8_epi32(c_int, 1),
                         _mm512_extracti32x8_epi32(d_int, 1)),
      _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i abcd_int8_low = _mm256_permute4x64_epi64(
      _mm256_packs_epi16(ab_int16_low, cd_int16_low), _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i abcd_int8_high =
      _mm256_permute4x64_epi64(_mm256_packs_epi16(ab_int16_high, cd_int16_high),
                               _MM_SHUFFLE(0, 2, 1, 3));
  return _mm512_inserti32x8(_mm512_castsi256_si512(abcd_int8_low),
                            abcd_int8_high, 1);
#endif
}

template <>
struct type_casting_traits<QInt32, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<QInt32, QInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8i
pcast<Packet16q32i, Packet64q8i>(const Packet16q32i& a, const Packet16q32i& b,
                                 const Packet16q32i& c, const Packet16q32i& d) {
  __m128i a_part = _mm512_cvtsepi32_epi8(a);
  __m128i b_part = _mm512_cvtsepi32_epi8(b);
  __m128i c_part = _mm512_cvtsepi32_epi8(c);
  __m128i d_part = _mm512_cvtsepi32_epi8(d);
  __m256i ab =
      _mm256_inserti128_si256(_mm256_castsi128_si256(a_part), b_part, 1);
  __m256i cd =
      _mm256_inserti128_si256(_mm256_castsi128_si256(c_part), d_part, 1);
  __m512i converted = _mm512_inserti64x4(_mm512_castsi256_si512(ab), cd, 1);
  return converted;
}

template <>
EIGEN_STRONG_INLINE Packet32q16i pcast<Packet16q32i, Packet32q16i>(
    const Packet16q32i& a, const Packet16q32i& b) {
  __m256i a_part = _mm512_cvtsepi32_epi16(a);
  __m256i b_part = _mm512_cvtsepi32_epi16(b);
  __m512i converted =
      _mm512_inserti64x4(_mm512_castsi256_si512(a_part), b_part, 1);
  return converted;
}

template <>
struct type_casting_traits<QInt32, QUInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8u
pcast<Packet16q32i, Packet64q8u>(const Packet16q32i& a, const Packet16q32i& b,
                                 const Packet16q32i& c, const Packet16q32i& d) {
  // Brute-force saturation since there isn't a pack operation for unsigned
  // numbers that keeps the elements in order.
  __m128i a_part = _mm512_cvtepi32_epi8(_mm512_max_epi32(
      _mm512_min_epi32(a, _mm512_set1_epi32(255)), _mm512_setzero_si512()));
  __m128i b_part = _mm512_cvtepi32_epi8(_mm512_max_epi32(
      _mm512_min_epi32(b, _mm512_set1_epi32(255)), _mm512_setzero_si512()));
  __m128i c_part = _mm512_cvtepi32_epi8(_mm512_max_epi32(
      _mm512_min_epi32(c, _mm512_set1_epi32(255)), _mm512_setzero_si512()));
  __m128i d_part = _mm512_cvtepi32_epi8(_mm512_max_epi32(
      _mm512_min_epi32(d, _mm512_set1_epi32(255)), _mm512_setzero_si512()));
  __m256i ab =
      _mm256_inserti128_si256(_mm256_castsi128_si256(a_part), b_part, 1);
  __m256i cd =
      _mm256_inserti128_si256(_mm256_castsi128_si256(c_part), d_part, 1);
  __m512i converted = _mm512_inserti64x4(_mm512_castsi256_si512(ab), cd, 1);
  return converted;
}

#if 0
// The type Packet32q16u does not exist for AVX-512 yet
template <>
struct type_casting_traits<QInt32, QUInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q16u
pcast<Packet16q32i, Packet32q16u>(const Packet16q32i& a,
                                  const Packet16q32i& b) {
  // Brute-force saturation since there isn't a pack operation for unsigned
  // numbers that keeps the elements in order.
  __m256i a_part =
      _mm512_cvtepi32_epi16(_mm512_max_epi32(
        _mm512_min_epi32(a, _mm512_set1_epi32(65535)), _mm512_setzero_si512()));
  __m256i b_part = _mm512_cvtepi32_epi16(
    _mm512_max_epi32(_mm512_min_epi32(b, _mm512_set1_epi32(65535)),
                     _mm512_setzero_si512()));
  __m512i converted =
      _mm512_inserti64x4(_mm512_castsi256_si512(a_part), b_part, 1);
  return converted;
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_
