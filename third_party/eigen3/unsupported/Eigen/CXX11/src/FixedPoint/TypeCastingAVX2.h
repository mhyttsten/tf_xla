#ifndef CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
#define CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX2DTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX2DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX2DTh() {
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

typedef __m256 Packet8f;

template <>
struct type_casting_traits<QInt32, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8q32i>(const Packet8q32i& a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX2DTh mht_0(mht_0_v, 184, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX2.h", "pcast<Packet8q32i>");

  return _mm256_cvtepi32_ps(a.m_val);
}

template <>
struct type_casting_traits<float, QInt32> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet8q32i pcast<Packet8f>(const Packet8f& a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSTypeCastingAVX2DTh mht_1(mht_1_v, 197, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/TypeCastingAVX2.h", "pcast<Packet8f>");

  return _mm256_cvtps_epi32(a);
}

template <>
struct type_casting_traits<QInt32, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8i
pcast<Packet8q32i, Packet32q8i>(const Packet8q32i& a, const Packet8q32i& b,
                                const Packet8q32i& c, const Packet8q32i& d) {
  __m256i converted = _mm256_packs_epi16(_mm256_packs_epi32(a.m_val, b.m_val),
                                         _mm256_packs_epi32(c.m_val, d.m_val));
  // Since packs does not cross 128 bit lane boundaries,
  // we have to permute to properly order the final result.
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

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
  __m256i converted = _mm256_packs_epi16(_mm256_packs_epi32(a_conv, b_conv),
                                         _mm256_packs_epi32(c_conv, d_conv));
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

template <>
struct type_casting_traits<QInt32, QUInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8u
pcast<Packet8q32i, Packet32q8u>(const Packet8q32i& a, const Packet8q32i& b,
                                const Packet8q32i& c, const Packet8q32i& d) {
  // _mm256_packus_epi32 trims negative numbers to 0 but we can't allow numbers
  // that are too large because _mm256_packus_epi16 expects signed input
  // (example of problem input: 0x11111111, which saturates to 0xffff = -1,
  // which saturates to 0).
  const __m256i a_clip = _mm256_min_epi32(a, _mm256_set1_epi32(255));
  const __m256i b_clip = _mm256_min_epi32(b, _mm256_set1_epi32(255));
  const __m256i c_clip = _mm256_min_epi32(c, _mm256_set1_epi32(255));
  const __m256i d_clip = _mm256_min_epi32(d, _mm256_set1_epi32(255));
  const __m256i converted = _mm256_packus_epi16(
      _mm256_packus_epi32(a_clip, b_clip), _mm256_packus_epi32(c_clip, d_clip));
  // Since packus does not cross 128 bit lane boundaries,
  // we have to permute to properly order the final result.
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
