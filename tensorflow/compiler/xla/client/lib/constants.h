/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONSTANTS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONSTANTS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh() {
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


#include <type_traits>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Returns scalar 'value' as a scalar of 'type'. Unlike ConstantR0, 'type' is
// determined at C++ run-time, rather than C++ compile-time.
// If 'value' is floating point but 'type' is not, or if 'value' is complex but
// 'type' is not, an error will be returned. This is to catch accidental
// truncation; in such cases, use an explicit cast.
template <typename T>
XlaOp ConstantR0WithType(XlaBuilder* builder, PrimitiveType type, T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/client/lib/constants.h", "ConstantR0WithType");

  if (std::is_floating_point<T>::value &&
      !(primitive_util::IsFloatingPointType(type) ||
        primitive_util::IsComplexType(type))) {
    return builder->ReportError(InvalidArgument(
        "Invalid cast from floating point type to %s in ConstantR0WithType.",
        PrimitiveType_Name(type)));
  }
  if (std::is_same<T, complex64>::value &&
      !primitive_util::IsComplexType(type)) {
    return builder->ReportError(InvalidArgument(
        "Invalid cast from complex type to %s in ConstantR0WithType.",
        PrimitiveType_Name(type)));
  }
  switch (type) {
    case PRED:
      return ConstantR0<bool>(builder, static_cast<bool>(value));
    case F16:
      return ConstantR0<half>(builder, static_cast<half>(value));
    case BF16:
      return ConstantR0<bfloat16>(builder, static_cast<bfloat16>(value));
    case F32:
      return ConstantR0<float>(builder, static_cast<float>(value));
    case F64:
      return ConstantR0<double>(builder, static_cast<double>(value));
    case C64:
      return ConstantR0<complex64>(builder, static_cast<complex64>(value));
    case C128:
      return ConstantR0<complex128>(builder, static_cast<complex128>(value));
    case U8:
      return ConstantR0<uint8_t>(builder, static_cast<uint8_t>(value));
    case U16:
      return ConstantR0<uint16_t>(builder, static_cast<uint16_t>(value));
    case U32:
      return ConstantR0<uint32_t>(builder, static_cast<uint32_t>(value));
    case U64:
      return ConstantR0<uint64_t>(builder, static_cast<uint64_t>(value));
    case S8:
      return ConstantR0<int8_t>(builder, static_cast<int8_t>(value));
    case S16:
      return ConstantR0<int16_t>(builder, static_cast<int16_t>(value));
    case S32:
      return ConstantR0<int32_t>(builder, static_cast<int32_t>(value));
    case S64:
      return ConstantR0<int64_t>(builder, static_cast<int64_t>(value));
    default:
      return builder->ReportError(
          InvalidArgument("Invalid type for ConstantR0WithType (%s).",
                          PrimitiveType_Name(type)));
  }
}

// Returns a scalar containing 'value' cast to the same run-time type as
// 'prototype'.
// If 'value' is floating point but 'prototype' is not, or if 'value' is complex
// 'prototype' is not, an error will be returned.
template <typename T>
XlaOp ScalarLike(XlaOp prototype, T value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh mht_1(mht_1_v, 263, "", "./tensorflow/compiler/xla/client/lib/constants.h", "ScalarLike");

  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return ConstantR0WithType(builder, shape.element_type(), value);
  });
}

// Returns an array or scalar containing copies of `value` cast to the same
// run-type type as `prototype` and broadcast to the same dimensions as
// `prototype`.
//
// If `prototype` is not a scalar or array, returns an error.
template <typename T>
XlaOp FullLike(XlaOp prototype, T value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTh mht_2(mht_2_v, 280, "", "./tensorflow/compiler/xla/client/lib/constants.h", "FullLike");

  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    if (ShapeUtil::IsScalar(shape) || shape.IsArray()) {
      return Broadcast(ScalarLike(prototype, value), shape.dimensions());
    } else {
      return InvalidArgument(
          "Prototype shape for BroadcastConstantLike must be a scalar or "
          "array, but was %s",
          shape.ToString());
    }
  });
}

// Returns a scalar with value '0' of 'type'.
XlaOp Zero(XlaBuilder* builder, PrimitiveType type);

// Returns a zero-filled tensor with shape `shape`.
XlaOp Zeros(XlaBuilder* builder, const Shape& shape);

// Returns a zero-filled tensor with the same shape as `prototype`.
XlaOp ZerosLike(XlaOp prototype);

// Returns a scalar with value '1' of 'type'.
XlaOp One(XlaBuilder* builder, PrimitiveType type);

// Returns the machine epsilon for floating-point type `type`, i.e.,
// the difference between 1.0 and the next representable value.
XlaOp Epsilon(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum representable finite or infinite value for 'type'.
// Returns '-inf' for floating-point types.
XlaOp MinValue(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum representable finite value for 'type'. For a floating
// point type, this is equal to -MaxFiniteValue().
XlaOp MinFiniteValue(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum positive normal value for floating-point type `type`.
XlaOp MinPositiveNormalValue(XlaBuilder* builder, PrimitiveType type);

// Returns the maximum representable finite or infinite value for 'type'.
// Returns 'inf' for floating-point types.
XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type);

// Returns the maximum representable finite value for 'type'.
XlaOp MaxFiniteValue(XlaBuilder* builder, PrimitiveType type);

// Returns a nan for the given type.  Only valid for real-valued fp types.
XlaOp NanValue(XlaBuilder* builder, PrimitiveType type);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONSTANTS_H_
