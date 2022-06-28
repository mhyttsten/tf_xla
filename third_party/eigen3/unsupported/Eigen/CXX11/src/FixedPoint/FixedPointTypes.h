// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_
#define CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_
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
class MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh {
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
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh() {
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


#include <cmath>
#include <iostream>

namespace Eigen {

// The mantissa part of the fixed point representation. See
// go/tensorfixedpoint for details
struct QInt8;
struct QUInt8;
struct QInt16;
struct QUInt16;
struct QInt32;

template <>
struct NumTraits<QInt8> : GenericNumTraits<int8_t> {};
template <>
struct NumTraits<QUInt8> : GenericNumTraits<uint8_t> {};
template <>
struct NumTraits<QInt16> : GenericNumTraits<int16_t> {};
template <>
struct NumTraits<QUInt16> : GenericNumTraits<uint16_t> {};
template <>
struct NumTraits<QInt32> : GenericNumTraits<int32_t> {};

namespace internal {
template <>
struct scalar_product_traits<QInt32, double> {
  enum {
    // Cost = NumTraits<T>::MulCost,
    Defined = 1
  };
  typedef QInt32 ReturnType;
};
}

// Wrap the 8bit int into a QInt8 struct instead of using a typedef to prevent
// the compiler from silently type cast the mantissa into a bigger or a smaller
// representation.
struct QInt8 {
  QInt8() : value(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_0(mht_0_v, 221, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt8");
}
  QInt8(const int8_t v) : value(v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_1(mht_1_v, 225, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt8");
}
  QInt8(const QInt32 v);

  operator int() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_2(mht_2_v, 231, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "int");
 return static_cast<int>(value); }

  int8_t value;
};

struct QUInt8 {
  QUInt8() : value(0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_3(mht_3_v, 240, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt8");
}
  QUInt8(const uint8_t v) : value(v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_4(mht_4_v, 244, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt8");
}
  QUInt8(const QInt32 v);

  operator int() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_5(mht_5_v, 250, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "int");
 return static_cast<int>(value); }

  uint8_t value;
};

struct QInt16 {
  QInt16() : value(0) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_6(mht_6_v, 259, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt16");
}
  QInt16(const int16_t v) : value(v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_7(mht_7_v, 263, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt16");
}
  QInt16(const QInt32 v);
  operator int() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_8(mht_8_v, 268, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "int");
 return static_cast<int>(value); }

  int16_t value;
};

struct QUInt16 {
  QUInt16() : value(0) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_9(mht_9_v, 277, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt16");
}
  QUInt16(const uint16_t v) : value(v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_10(mht_10_v, 281, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt16");
}
  QUInt16(const QInt32 v);
  operator int() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_11(mht_11_v, 286, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "int");
 return static_cast<int>(value); }

  uint16_t value;
};

struct QInt32 {
  QInt32() : value(0) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_12(mht_12_v, 295, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
  QInt32(const int8_t v) : value(v) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_13(mht_13_v, 299, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
  QInt32(const int32_t v) : value(v) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_14(mht_14_v, 303, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
  QInt32(const uint32_t v) : value(static_cast<int32_t>(v)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_15(mht_15_v, 307, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
  QInt32(const QInt8 v) : value(v.value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_16(mht_16_v, 311, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
  QInt32(const float v) : value(static_cast<int32_t>(lrint(v))) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_17(mht_17_v, 315, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt32");
}
#ifdef EIGEN_MAKING_DOCS
  // Workaround to fix build on PPC.
  QInt32(unsigned long v) : value(v) {}
#endif

  operator float() const { return static_cast<float>(value); }

  int32_t value;
};

EIGEN_STRONG_INLINE QInt8::QInt8(const QInt32 v)
    : value(v.value > 127 ? 127 : (v.value < -128 ? -128 : v.value)) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_18(mht_18_v, 330, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt8::QInt8");
}
EIGEN_STRONG_INLINE QUInt8::QUInt8(const QInt32 v)
    : value(v.value > 255 ? 255 : (v.value < 0 ? 0 : v.value)) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_19(mht_19_v, 335, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt8::QUInt8");
}
EIGEN_STRONG_INLINE QInt16::QInt16(const QInt32 v)
    : value(v.value > 32767 ? 32767 : (v.value < -32768 ? -32768 : v.value)) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_20(mht_20_v, 340, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QInt16::QInt16");
}
EIGEN_STRONG_INLINE QUInt16::QUInt16(const QInt32 v)
    : value(v.value > 65535 ? 65535 : (v.value < 0 ? 0 : v.value)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_21(mht_21_v, 345, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "QUInt16::QUInt16");
}

// Basic widening 8-bit operations: This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QInt8 b) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_22(mht_22_v, 351, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QUInt8 b) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_23(mht_23_v, 357, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt8 a, const QInt8 b) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_24(mht_24_v, 363, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt8 a, const QInt8 b) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_25(mht_25_v, 369, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - static_cast<int32_t>(b.value));
}

// Basic widening 16-bit operations: This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QInt16 b) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_26(mht_26_v, 377, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QUInt16 b) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_27(mht_27_v, 383, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt16 a, const QInt16 b) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_28(mht_28_v, 389, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt16 a, const QInt16 b) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_29(mht_29_v, 395, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - static_cast<int32_t>(b.value));
}

// Mixed QInt32 op QInt8 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt8 b) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_30(mht_30_v, 403, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt8 a, const QInt32 b) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_31(mht_31_v, 409, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt8 b) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_32(mht_32_v, 415, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt8 a, const QInt32 b) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_33(mht_33_v, 421, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt8 b) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_34(mht_34_v, 427, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt8 a, const QInt32 b) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_35(mht_35_v, 433, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QInt16 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt16 b) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_36(mht_36_v, 441, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QInt16 a, const QInt32 b) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_37(mht_37_v, 447, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt16 b) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_38(mht_38_v, 453, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt16 a, const QInt32 b) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_39(mht_39_v, 459, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt16 b) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_40(mht_40_v, 465, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt16 a, const QInt32 b) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_41(mht_41_v, 471, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QUInt8 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QUInt8 b) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_42(mht_42_v, 479, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QUInt8 a, const QInt32 b) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_43(mht_43_v, 485, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QUInt8 b) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_44(mht_44_v, 491, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QUInt8 a, const QInt32 b) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_45(mht_45_v, 497, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QUInt8 b) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_46(mht_46_v, 503, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QUInt8 a, const QInt32 b) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_47(mht_47_v, 509, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Mixed QInt32 op QUInt16 operations. This will be vectorized in future CLs.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QUInt16 b) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_48(mht_48_v, 517, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(a.value + static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator+(const QUInt16 a, const QInt32 b) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_49(mht_49_v, 523, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return QInt32(static_cast<int32_t>(a.value) + b.value);
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QUInt16 b) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_50(mht_50_v, 529, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(a.value - static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator-(const QUInt16 a, const QInt32 b) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_51(mht_51_v, 535, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return QInt32(static_cast<int32_t>(a.value) - b.value);
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QUInt16 b) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_52(mht_52_v, 541, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(a.value * static_cast<int32_t>(b.value));
}
EIGEN_STRONG_INLINE QInt32 operator*(const QUInt16 a, const QInt32 b) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_53(mht_53_v, 547, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return QInt32(static_cast<int32_t>(a.value) * b.value);
}

// Basic arithmetic operations on QInt32, which behaves like a int32_t.
EIGEN_STRONG_INLINE QInt32 operator+(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_54(mht_54_v, 555, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "+");

  return a.value + b.value;
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_55(mht_55_v, 561, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");

  return a.value - b.value;
}
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_56(mht_56_v, 567, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return a.value * b.value;
}
EIGEN_STRONG_INLINE QInt32 operator/(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_57(mht_57_v, 573, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "/");

  return a.value / b.value;
}
EIGEN_STRONG_INLINE QInt32& operator+=(QInt32& a, const QInt32 b) {
  a.value += b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator-=(QInt32& a, const QInt32 b) {
  a.value -= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator*=(QInt32& a, const QInt32 b) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_58(mht_58_v, 587, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "=");

  a.value *= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32& operator/=(QInt32& a, const QInt32 b) {
  a.value /= b.value;
  return a;
}
EIGEN_STRONG_INLINE QInt32 operator-(const QInt32 a) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_59(mht_59_v, 598, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "-");
 return -a.value; }

// Scaling QInt32 by double. We do the arithmetic in double because
// float only has 23 bits of mantissa, so casting QInt32 to float might reduce
// accuracy by discarding up to 7 (least significant) bits.
EIGEN_STRONG_INLINE QInt32 operator*(const QInt32 a, const double b) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_60(mht_60_v, 606, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return static_cast<int32_t>(lrint(static_cast<double>(a.value) * b));
}
EIGEN_STRONG_INLINE QInt32 operator*(const double a, const QInt32 b) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_61(mht_61_v, 612, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "*");

  return static_cast<int32_t>(lrint(a * static_cast<double>(b.value)));
}
EIGEN_STRONG_INLINE QInt32& operator*=(QInt32& a, const double b) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_62(mht_62_v, 618, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "=");

  a.value = static_cast<int32_t>(lrint(static_cast<double>(a.value) * b));
  return a;
}

// Comparisons
EIGEN_STRONG_INLINE bool operator==(const QInt8 a, const QInt8 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QUInt8 a, const QUInt8 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QInt16 a, const QInt16 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QUInt16 a, const QUInt16 b) {
  return a.value == b.value;
}
EIGEN_STRONG_INLINE bool operator==(const QInt32 a, const QInt32 b) {
  return a.value == b.value;
}

EIGEN_STRONG_INLINE bool operator<(const QInt8 a, const QInt8 b) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_63(mht_63_v, 643, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<");

  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QUInt8 a, const QUInt8 b) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_64(mht_64_v, 649, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<");

  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QInt16 a, const QInt16 b) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_65(mht_65_v, 655, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<");

  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QUInt16 a, const QUInt16 b) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_66(mht_66_v, 661, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<");

  return a.value < b.value;
}
EIGEN_STRONG_INLINE bool operator<(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_67(mht_67_v, 667, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<");

  return a.value < b.value;
}

EIGEN_STRONG_INLINE bool operator>(const QInt8 a, const QInt8 b) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_68(mht_68_v, 674, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator>");

  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QUInt8 a, const QUInt8 b) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_69(mht_69_v, 680, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator>");

  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QInt16 a, const QInt16 b) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_70(mht_70_v, 686, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator>");

  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QUInt16 a, const QUInt16 b) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_71(mht_71_v, 692, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator>");

  return a.value > b.value;
}
EIGEN_STRONG_INLINE bool operator>(const QInt32 a, const QInt32 b) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_72(mht_72_v, 698, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator>");

  return a.value > b.value;
}

EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt8 a) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_73(mht_73_v, 705, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<<");

  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QUInt8 a) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_74(mht_74_v, 712, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<<");

  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt16 a) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_75(mht_75_v, 719, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<<");

  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QUInt16 a) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_76(mht_76_v, 726, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<<");

  os << static_cast<int>(a.value);
  return os;
}
EIGEN_STRONG_INLINE std::ostream& operator<<(std::ostream& os, QInt32 a) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPSthird_partyPSeigen3PSunsupportedPSEigenPSCXX11PSsrcPSFixedPointPSFixedPointTypesDTh mht_77(mht_77_v, 733, "", "./third_party/eigen3/unsupported/Eigen/CXX11/src/FixedPoint/FixedPointTypes.h", "operator<<");

  os << a.value;
  return os;
}

}  // namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_FIXEDPOINTTYPES_H_
