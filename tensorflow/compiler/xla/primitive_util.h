/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Utilities for dealing with XLA primitive types.

#ifndef TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh() {
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


#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace primitive_util {

// Returns the count of significand (mantissa) bits for float datatypes.
// For non-float datatypes, results in a LOG(FATAL).
int SignificandWidth(PrimitiveType type);

// Returns the count of exponent bits for float datatypes.
// For non-float datatypes, results in a LOG(FATAL).
int ExponentWidth(PrimitiveType type);

// Returns the exponent of the smallest number which cannot be represented.
// For non-float datatypes, results in a LOG(FATAL).
int OverflowExponent(PrimitiveType type);

// Returns the XLA primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
PrimitiveType NativeToPrimitiveType() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType");

  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PRIMITIVE_TYPE_INVALID;
}

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.  As an optimization, these are declared inline in the
// header.
template <>
inline PrimitiveType NativeToPrimitiveType<bool>() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_1(mht_1_v, 234, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<bool>");

  return PRED;
}

// Unsigned integer
template <>
inline PrimitiveType NativeToPrimitiveType<uint8_t>() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<uint8_t>");

  return U8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint16_t>() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<uint16_t>");

  return U16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint32_t>() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_4(mht_4_v, 259, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<uint32_t>");

  return U32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint64_t>() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_5(mht_5_v, 267, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<uint64_t>");

  return U64;
}

// Signed integer
template <>
inline PrimitiveType NativeToPrimitiveType<int8_t>() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_6(mht_6_v, 276, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<int8_t>");

  return S8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int16_t>() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_7(mht_7_v, 284, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<int16_t>");

  return S16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int32_t>() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_8(mht_8_v, 292, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<int32_t>");

  return S32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int64_t>() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_9(mht_9_v, 300, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<int64_t>");

  return S64;
}

// Floating point
template <>
inline PrimitiveType NativeToPrimitiveType<float>() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_10(mht_10_v, 309, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<float>");

  return F32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<double>() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_11(mht_11_v, 317, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<double>");

  return F64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<half>() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_12(mht_12_v, 325, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<half>");

  return F16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<bfloat16>() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_13(mht_13_v, 333, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<bfloat16>");

  return BF16;
}

// Complex
template <>
inline PrimitiveType NativeToPrimitiveType<complex64>() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_14(mht_14_v, 342, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<complex64>");

  return C64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<complex128>() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_15(mht_15_v, 350, "", "./tensorflow/compiler/xla/primitive_util.h", "NativeToPrimitiveType<complex128>");

  return C128;
}

bool IsFloatingPointType(PrimitiveType type);

bool IsComplexType(PrimitiveType type);

bool IsSignedIntegralType(PrimitiveType type);

bool IsUnsignedIntegralType(PrimitiveType type);

bool IsIntegralType(PrimitiveType type);

// Returns true if values of the given primitive type are held in array shapes.
bool IsArrayType(PrimitiveType primitive_type);

// Returns the number of bits in the representation for a given type.
int BitWidth(PrimitiveType type);

// Returns the number of bytes in the representation for a given type.
int ByteWidth(PrimitiveType type);

PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth);

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth);

// Returns the real, imag component type underlying the given complex type.
// LOG(FATAL)'s if complex_type is not complex.
PrimitiveType ComplexComponentType(PrimitiveType complex_type);

// Returns the higher-precision element type if a and b are both floating
// point types; otherwise, checks that they have the same element type
// and returns it.
inline PrimitiveType HigherPrecisionType(PrimitiveType a, PrimitiveType b) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_16(mht_16_v, 387, "", "./tensorflow/compiler/xla/primitive_util.h", "HigherPrecisionType");

  // Returns a tuple where the elements are lexicographically ordered in terms
  // of importance.
  auto type_properties = [](PrimitiveType type) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_17(mht_17_v, 393, "", "./tensorflow/compiler/xla/primitive_util.h", "lambda");

    auto component_type =
        IsComplexType(type) ? ComplexComponentType(type) : type;
    return std::make_tuple(
        // Prefer complex types over non-complex types.
        IsComplexType(type),
        // Prefer floating point types with more range over other
        // floating-point types or non-floating point types.
        IsFloatingPointType(component_type) ? OverflowExponent(component_type)
                                            : -1,
        // Prefer floating point types with more precision over less precise
        // types.
        IsFloatingPointType(component_type) ? SignificandWidth(component_type)
                                            : -1,
        // Prefer wider types over narrower types.
        BitWidth(component_type),
        // Prefer signed integer types over unsigned integer types.
        IsSignedIntegralType(component_type));
  };
  auto a_properties = type_properties(a);
  auto b_properties = type_properties(b);
  if (a_properties > b_properties) {
    return a;
  }
  if (b_properties > a_properties) {
    return b;
  }
  CHECK_EQ(a, b);
  return a;
}

// Returns true if a convert from from_type to to_type loses no precision.
inline bool CastPreservesValues(PrimitiveType from_type,
                                PrimitiveType to_type) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTh mht_18(mht_18_v, 429, "", "./tensorflow/compiler/xla/primitive_util.h", "CastPreservesValues");

  // * -> *
  if (from_type == to_type) {
    return true;
  }
  // PRED -> *
  if (from_type == PRED) {
    return true;
  }
  // ~PRED -> PRED is not safe because it drops almost all numbers.
  if (to_type == PRED) {
    return false;
  }
  // * -> C is safe if the components of * and C can be safely converted.
  if (primitive_util::IsComplexType(to_type)) {
    auto from_component_type =
        primitive_util::IsComplexType(from_type)
            ? primitive_util::ComplexComponentType(from_type)
            : from_type;
    auto to_component_type = primitive_util::ComplexComponentType(to_type);
    return CastPreservesValues(from_component_type, to_component_type);
  }
  // ~C -> C is not safe because it drops imaginary components.
  if (primitive_util::IsComplexType(from_type)) {
    return false;
  }
  // F -> F is safe if the exponent and significand are preserved.
  if (primitive_util::IsFloatingPointType(from_type) &&
      primitive_util::IsFloatingPointType(to_type)) {
    return primitive_util::SignificandWidth(from_type) <=
               primitive_util::SignificandWidth(to_type) &&
           primitive_util::ExponentWidth(from_type) <=
               primitive_util::ExponentWidth(to_type) &&
           primitive_util::OverflowExponent(from_type) <=
               primitive_util::OverflowExponent(to_type);
  }
  // F -> I is not safe because it drops fractional numbers.
  if (!primitive_util::IsIntegralType(from_type)) {
    return false;
  }
  // An n-bit unsigned integer takes on values from [0, 2^n - 1].
  // An n-bit signed integer takes on values from [-2^(n-1), 2^(n-1) - 1].
  // from_bits/to_bits considers the number of non-sign bits.
  const int from_bits = primitive_util::IsSignedIntegralType(from_type)
                            ? primitive_util::BitWidth(from_type) - 1
                            : primitive_util::BitWidth(from_type);
  const int to_bits = primitive_util::IsSignedIntegralType(to_type)
                          ? primitive_util::BitWidth(to_type) - 1
                          : primitive_util::BitWidth(to_type);
  // I -> F is safe if the integer can be represented exactly.
  if (primitive_util::IsFloatingPointType(to_type)) {
    // In both cases, we need to handle an exponent of n-1.
    // However, the significand needed to represent signed two's complement
    // numbers is smaller by one bit because it will only have a non-zero
    // trailing significand field when the exponent is smaller than n-1.
    return from_bits <= primitive_util::SignificandWidth(to_type) &&
           primitive_util::BitWidth(from_type) - 1 <
               primitive_util::OverflowExponent(to_type);
  }
  // S -> U is not safe because it drops negative numbers.
  if (primitive_util::IsSignedIntegralType(from_type) &&
      primitive_util::IsUnsignedIntegralType(to_type)) {
    return false;
  }
  // I -> I is safe if the integer can be represented exactly; we've already
  // ensured that signed to unsigned conversions won't happen here.
  CHECK(primitive_util::IsIntegralType(to_type));
  return from_bits <= to_bits;
}

// Returns the native type (eg, float) corresponding to the given template
// parameter XLA primitive type (eg, F32).
template <PrimitiveType>
struct PrimitiveTypeToNative;

// Declarations of specializations for each native type which correspond to a
// XLA primitive type.
template <>
struct PrimitiveTypeToNative<PRED> {
  using type = bool;
};

// Unsigned integer
template <>
struct PrimitiveTypeToNative<U8> {
  using type = uint8_t;
};

template <>
struct PrimitiveTypeToNative<U16> {
  using type = uint16_t;
};

template <>
struct PrimitiveTypeToNative<U32> {
  using type = uint32_t;
};

template <>
struct PrimitiveTypeToNative<U64> {
  using type = uint64_t;
};

// Signed integer
template <>
struct PrimitiveTypeToNative<S8> {
  using type = int8_t;
};

template <>
struct PrimitiveTypeToNative<S16> {
  using type = int16_t;
};

template <>
struct PrimitiveTypeToNative<S32> {
  using type = int32_t;
};

template <>
struct PrimitiveTypeToNative<S64> {
  using type = int64_t;
};

// Floating point
template <>
struct PrimitiveTypeToNative<F32> {
  using type = float;
};
template <>
struct PrimitiveTypeToNative<F64> {
  using type = double;
};
template <>
struct PrimitiveTypeToNative<F16> {
  using type = half;
};

template <>
struct PrimitiveTypeToNative<BF16> {
  using type = bfloat16;
};

// Complex
template <>
struct PrimitiveTypeToNative<C64> {
  using type = complex64;
};

template <>
struct PrimitiveTypeToNative<C128> {
  using type = complex128;
};

// Returns the lower-case name of the given primitive type.
const std::string& LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name);

// Returns true if the given name is a primitive type string (lower-case).
bool IsPrimitiveTypeName(absl::string_view name);

}  // namespace primitive_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PRIMITIVE_UTIL_H_
