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
class MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc() {
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

#include "tensorflow/compiler/xla/primitive_util.h"

#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace primitive_util {

int SignificandWidth(PrimitiveType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/primitive_util.cc", "SignificandWidth");

  switch (type) {
    case F32:
      return std::numeric_limits<float>::digits;
    case F64:
      return std::numeric_limits<double>::digits;
    case BF16:
      return std::numeric_limits<bfloat16>::digits;
    case F16:
      return std::numeric_limits<half>::digits;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
}

int ExponentWidth(PrimitiveType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/primitive_util.cc", "ExponentWidth");

  // Per the IEEE-754 standard: a floating point type is stored as a sign bit, a
  // biased exponent and a trailing significand field.
  int total_bit_width = BitWidth(type);
  // This field contains all bits in the significand other than the leading
  // digit which is implied by the exponent.
  int trailing_significand_field_width = SignificandWidth(type) - 1;
  // The sign is encoded with a single bit.
  int kSignBitWidth = 1;
  // The remaining bits are used for encoding the biased exponent.
  return total_bit_width - (trailing_significand_field_width + kSignBitWidth);
}

int OverflowExponent(PrimitiveType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/primitive_util.cc", "OverflowExponent");

  // |std::numeric_limits<float>::max_exponent| is defined as: "Maximum positive
  // integer such that radix raised to the power one less than that integer is a
  // representable finite floating-point number." as such it does not actually
  // yield the maximum exponent but the exponent of the first integer which
  // overflows.
  switch (type) {
    case F32:
      return std::numeric_limits<float>::max_exponent;
    case F64:
      return std::numeric_limits<double>::max_exponent;
    case BF16:
      return std::numeric_limits<bfloat16>::max_exponent;
    case F16:
      return std::numeric_limits<half>::max_exponent;
    default:
      LOG(FATAL) << "Not a floating data type " << type;
  }
}

bool IsFloatingPointType(PrimitiveType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsFloatingPointType");

  return type == F16 || type == F32 || type == F64 || type == BF16;
}

bool IsComplexType(PrimitiveType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsComplexType");
 return type == C64 || type == C128; }

bool IsSignedIntegralType(PrimitiveType type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_5(mht_5_v, 269, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsSignedIntegralType");

  return type == S8 || type == S16 || type == S32 || type == S64;
}

bool IsUnsignedIntegralType(PrimitiveType type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_6(mht_6_v, 276, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsUnsignedIntegralType");

  return type == U8 || type == U16 || type == U32 || type == U64;
}

bool IsIntegralType(PrimitiveType type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_7(mht_7_v, 283, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsIntegralType");

  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

int BitWidth(PrimitiveType type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_8(mht_8_v, 290, "", "./tensorflow/compiler/xla/primitive_util.cc", "BitWidth");

  switch (type) {
    case PRED:
      return 1;

    case S8:
    case U8:
      return 8;

    case S16:
    case U16:
    case F16:
    case BF16:
      return 16;

    case U32:
    case S32:
    case F32:
      return 32;

    case U64:
    case S64:
    case F64:
    case C64:
      return 64;

    case C128:
      return 128;

    case TUPLE:
      LOG(FATAL) << "TUPLE is an invalid type for BitWidth";

    case OPAQUE_TYPE:
      LOG(FATAL) << "OPAQUE_TYPE is an invalid type for BitWidth";

    default:
      LOG(FATAL) << "Unhandled primitive type " << type;
  }
}

int ByteWidth(PrimitiveType type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_9(mht_9_v, 333, "", "./tensorflow/compiler/xla/primitive_util.cc", "ByteWidth");
 return CeilOfRatio(BitWidth(type), 8); }

xla::PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_10(mht_10_v, 338, "", "./tensorflow/compiler/xla/primitive_util.cc", "UnsignedIntegralTypeForBitWidth");

  switch (src_bitwidth) {
    case 8:
      return xla::U8;
    case 16:
      return xla::U16;
    case 32:
      return xla::U32;
    case 64:
      return xla::U64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

xla::PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_11(mht_11_v, 356, "", "./tensorflow/compiler/xla/primitive_util.cc", "SignedIntegralTypeForBitWidth");

  switch (src_bitwidth) {
    case 8:
      return xla::S8;
    case 16:
      return xla::S16;
    case 32:
      return xla::S32;
    case 64:
      return xla::S64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

PrimitiveType ComplexComponentType(PrimitiveType complex_type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_12(mht_12_v, 374, "", "./tensorflow/compiler/xla/primitive_util.cc", "ComplexComponentType");

  switch (complex_type) {
    case C64:
      return F32;
    case C128:
      return F64;
    default:
      LOG(FATAL) << "Primitive type is not complex: "
                 << PrimitiveType_Name(complex_type);
  }
}

bool IsArrayType(PrimitiveType primitive_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_13(mht_13_v, 389, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsArrayType");

  return primitive_type != PRIMITIVE_TYPE_INVALID && primitive_type != TUPLE &&
         primitive_type != OPAQUE_TYPE && primitive_type != TOKEN;
}

// Class to memoize the computation of
//   absl::AsciiStrToLower(PrimitiveType_Name(p))
// for all PrimitiveType values "p"
//
// xla::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_14(mht_14_v, 405, "", "./tensorflow/compiler/xla/primitive_util.cc", "PrimitiveTypeNameGenerator");

    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (i == static_cast<int>(OPAQUE_TYPE)) {
        lowercase_name_[i] = "opaque";
      } else if (PrimitiveType_IsValid(i)) {
        lowercase_name_[i] = absl::AsciiStrToLower(
            PrimitiveType_Name(static_cast<PrimitiveType>(i)));
      }
    }
  }
  const std::string& LowercaseName(PrimitiveType t) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_15(mht_15_v, 418, "", "./tensorflow/compiler/xla/primitive_util.cc", "LowercaseName");

    return lowercase_name_[static_cast<int>(t)];
  }

 private:
  std::string lowercase_name_[PrimitiveType_ARRAYSIZE];
};

const std::string& LowercasePrimitiveTypeName(PrimitiveType s) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_16(mht_16_v, 429, "", "./tensorflow/compiler/xla/primitive_util.cc", "LowercasePrimitiveTypeName");

  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the xla::OPAQUE_TYPE enumerator.
const absl::flat_hash_map<std::string, PrimitiveType>&
GetPrimitiveTypeStringMap() {
  static absl::flat_hash_map<std::string, PrimitiveType>* name_to_type = [] {
    static auto* map = new absl::flat_hash_map<std::string, PrimitiveType>;
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (PrimitiveType_IsValid(i) && i != PRIMITIVE_TYPE_INVALID) {
        auto value = static_cast<PrimitiveType>(i);
        (*map)[LowercasePrimitiveTypeName(value)] = value;
      }
    }
    (*map)["opaque"] = OPAQUE_TYPE;
    return map;
  }();
  return *name_to_type;
}

}  // namespace

StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(std::string(name));
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", name);
  }
  return found->second;
}

bool IsPrimitiveTypeName(absl::string_view name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSprimitive_utilDTcc mht_17(mht_17_v, 471, "", "./tensorflow/compiler/xla/primitive_util.cc", "IsPrimitiveTypeName");

  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(std::string(name));
  return found != map.end();
}

}  // namespace primitive_util
}  // namespace xla
