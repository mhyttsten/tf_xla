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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/constants.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp Zero(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_0(mht_0_v, 192, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "Zero");

  return ConstantLiteral(builder, LiteralUtil::Zero(type));
}

XlaOp Zeros(XlaBuilder* builder, const Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_1(mht_1_v, 199, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "Zeros");

  return Broadcast(Zero(builder, shape.element_type()), shape.dimensions());
}

XlaOp ZerosLike(XlaOp prototype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_2(mht_2_v, 206, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "ZerosLike");

  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return Zeros(builder, shape);
  });
}

XlaOp One(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_3(mht_3_v, 217, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "One");

  return ConstantLiteral(builder, LiteralUtil::One(type));
}

XlaOp Epsilon(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_4(mht_4_v, 224, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "Epsilon");

  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(
          builder,
          static_cast<Eigen::half>(Eigen::NumTraits<Eigen::half>::epsilon()));
    case BF16:
      return ConstantR0<Eigen::bfloat16>(
          builder, static_cast<Eigen::bfloat16>(
                       Eigen::NumTraits<Eigen::bfloat16>::epsilon()));
    case F32:
      return ConstantR0<float>(builder, std::numeric_limits<float>::epsilon());
    case F64:
      return ConstantR0<double>(builder,
                                std::numeric_limits<double>::epsilon());
    default:
      return builder->ReportError(InvalidArgument(
          "Invalid type for Epsilon (%s).", PrimitiveType_Name(type)));
  }
}

XlaOp MinValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_5(mht_5_v, 248, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "MinValue");

  return ConstantLiteral(builder, LiteralUtil::MinValue(type));
}

XlaOp MinFiniteValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_6(mht_6_v, 255, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "MinFiniteValue");

  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(builder,
                                     Eigen::NumTraits<Eigen::half>::lowest());
    case BF16:
      return ConstantR0<Eigen::bfloat16>(
          builder, Eigen::NumTraits<Eigen::bfloat16>::lowest());
    case F32:
      return ConstantR0<float>(builder, -std::numeric_limits<float>::max());
    case F64:
      return ConstantR0<double>(builder, -std::numeric_limits<double>::max());
    default:
      return MinValue(builder, type);
  }
}

XlaOp MinPositiveNormalValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_7(mht_7_v, 275, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "MinPositiveNormalValue");

  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(builder,
                                     std::numeric_limits<Eigen::half>::min());
    case BF16:
      return ConstantR0<Eigen::bfloat16>(
          builder, std::numeric_limits<Eigen::bfloat16>::min());
    case F32:
      return ConstantR0<float>(builder, std::numeric_limits<float>::min());
    case F64:
      return ConstantR0<double>(builder, std::numeric_limits<double>::min());
    default:
      return builder->ReportError(
          InvalidArgument("Invalid type for MinPositiveNormalValue (%s).",
                          PrimitiveType_Name(type)));
  }
}

XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_8(mht_8_v, 297, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "MaxValue");

  return ConstantLiteral(builder, LiteralUtil::MaxValue(type));
}

XlaOp MaxFiniteValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_9(mht_9_v, 304, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "MaxFiniteValue");

  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(builder,
                                     Eigen::NumTraits<Eigen::half>::highest());
    case BF16:
      return ConstantR0<Eigen::bfloat16>(
          builder, Eigen::NumTraits<Eigen::bfloat16>::highest());
    case F32:
      return ConstantR0<float>(builder, std::numeric_limits<float>::max());
    case F64:
      return ConstantR0<double>(builder, std::numeric_limits<double>::max());
    default:
      return MaxValue(builder, type);
  }
}

XlaOp NanValue(XlaBuilder* builder, PrimitiveType type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSconstantsDTcc mht_10(mht_10_v, 324, "", "./tensorflow/compiler/xla/client/lib/constants.cc", "NanValue");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    switch (type) {
      case F16:
        return ConstantR0<Eigen::half>(
            builder, Eigen::NumTraits<Eigen::half>::quiet_NaN());
      case BF16:
        return ConstantR0<Eigen::bfloat16>(
            builder, Eigen::NumTraits<Eigen::bfloat16>::quiet_NaN());
      case F32:
        return ConstantR0<float>(builder,
                                 std::numeric_limits<float>::quiet_NaN());
      case F64:
        return ConstantR0<double>(builder,
                                  std::numeric_limits<double>::quiet_NaN());
      default:
        return InvalidArgument(
            "Operand to NanValue was %s, but must be a real-valued "
            "floating-point type.",
            PrimitiveType_Name(type));
    }
  });
}

}  // namespace xla
