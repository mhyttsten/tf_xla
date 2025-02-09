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
class MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

// Cwise binary ops
Status GradForUnaryCwise(FunctionDef* g, std::vector<FDH::Node> nodes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/ops/math_grad.cc", "GradForUnaryCwise");

  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", "$T"}};
    }
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status AbsGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/ops/math_grad.cc", "AbsGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sign"}, "Sign", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "sign"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Abs", AbsGrad);

Status NegGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/ops/math_grad.cc", "NegGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Neg", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Neg", NegGrad);

Status InvGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/ops/math_grad.cc", "InvGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Reciprocal", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      {{"y2_neg"}, "Neg", {"y2"}},
      {{"dx"}, "Mul", {"dy", "y2_neg"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Inv", InvGrad);
REGISTER_OP_GRADIENT("Reciprocal", InvGrad);

Status SquareGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/ops/math_grad.cc", "SquareGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("c", int64_t{2}),
      {{"two"}, "Cast", {"c"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
      {{"x2"}, "Mul", {"x", "two"}, {}, {"dy"}},  // x * 2
      {{"dx"}, "Mul", {"dy", "x2"}},              // dy * (x * 2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Square", SquareGrad);

Status SqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/ops/math_grad.cc", "SqrtGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Sqrt", {"x"}},
      {{"y_inv"}, "Reciprocal", {"y"}, {}, {"dy"}},
      FDH::Const("const", 0.5f),
      {{"half"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Mul", {"half", "y_inv"}},  // .5 * 1/y
      {{"dx"}, "Mul", {"dy", "a"}},  // dy * (.5 * 1/y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sqrt", SqrtGrad);

Status RsqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/ops/math_grad.cc", "RsqrtGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Reciprocal", {"x"}, {}, {"dy"}},
      {{"y"}, "Rsqrt", {"x"}},
      FDH::Const("const", -.5f),
      {{"neghalf"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Mul", {"neghalf", "x_inv"}},   // -0.5 * 1/x
      {{"b"}, "Mul", {"a", "y"}},             // -0.5 * 1/x * y
      {{"dx"}, "Mul", {"dy", "b"}},           // dy * (1/y * .5)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Rsqrt", RsqrtGrad);

Status ExpGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_7(mht_7_v, 309, "", "./tensorflow/core/ops/math_grad.cc", "ExpGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Exp", {"x"}},
      {{"dx"}, "Mul", {"dy", "y"}},           // dy * y
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Exp", ExpGrad);

Status Expm1Grad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_8(mht_8_v, 322, "", "./tensorflow/core/ops/math_grad.cc", "Expm1Grad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Exp", {"x"}},
      {{"dx"}, "Mul", {"dy", "y"}},           // dy * y
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Expm1", Expm1Grad);

Status LogGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_9(mht_9_v, 335, "", "./tensorflow/core/ops/math_grad.cc", "LogGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Reciprocal", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "x_inv"}},           // dy * 1/x
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Log", LogGrad);

Status Log1pGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_10(mht_10_v, 348, "", "./tensorflow/core/ops/math_grad.cc", "Log1pGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Add", {"one", "x"}},
      {{"dx"}, "Div", {"dy", "a"}},           // dy / (1 + x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Log1p", Log1pGrad);

Status SinhGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_11(mht_11_v, 363, "", "./tensorflow/core/ops/math_grad.cc", "SinhGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"cosh"}, "Cosh", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "cosh"}},  // dy * cosh(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sinh", SinhGrad);

Status CoshGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_12(mht_12_v, 376, "", "./tensorflow/core/ops/math_grad.cc", "CoshGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sinh"}, "Sinh", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "sinh"}},  // dy * sinh(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Cosh", CoshGrad);

Status TanhGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_13(mht_13_v, 389, "", "./tensorflow/core/ops/math_grad.cc", "TanhGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Tanh", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Sub", {"one", "y2"}},
      {{"dx"}, "Mul", {"dy", "a"}},           // dy * (1 - y*y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Tanh", TanhGrad);

Status AsinhGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_14(mht_14_v, 406, "", "./tensorflow/core/ops/math_grad.cc", "AsinhGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Asinh", {"x"}},
      {{"cosh"}, "Cosh", {"y"}},
      {{"dx"}, "Mul", {"dy", "cosh"}},  // dy * cosh(y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Asinh", AsinhGrad);

Status AcoshGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_15(mht_15_v, 420, "", "./tensorflow/core/ops/math_grad.cc", "AcoshGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Acosh", {"x"}},
      {{"sinh"}, "Sinh", {"y"}},
      {{"dx"}, "Mul", {"dy", "sinh"}},  // dy * sinh(y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Acosh", AcoshGrad);

Status AtanhGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_16(mht_16_v, 434, "", "./tensorflow/core/ops/math_grad.cc", "AtanhGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"inv"}, "Reciprocal", {"a"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Atanh", AtanhGrad);

Status SigmoidGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_17(mht_17_v, 451, "", "./tensorflow/core/ops/math_grad.cc", "SigmoidGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Sigmoid", {"x"}},
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Sub", {"one", "y"}, {}, {"dy"}},
      {{"b"}, "Mul", {"y", "a"}},             // y * (1 - y)
      {{"dx"}, "Mul", {"dy", "b"}},           // dy * y * (1 - y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sigmoid", SigmoidGrad);

Status SignGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_18(mht_18_v, 468, "", "./tensorflow/core/ops/math_grad.cc", "SignGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"s"}, "Shape", {"x"}},
      FDH::Const("zero", 0.f),
      {{"val"}, "Cast", {"zero"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"dx"}, "Fill", {"s", "val"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sign", SignGrad);

Status SinGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_19(mht_19_v, 483, "", "./tensorflow/core/ops/math_grad.cc", "SinGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"cos"}, "Cos", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "cos"}},  // dy * cos(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sin", SinGrad);

Status CosGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_20(mht_20_v, 496, "", "./tensorflow/core/ops/math_grad.cc", "CosGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sin"}, "Sin", {"x"}, {}, {"dy"}},
      {{"neg"}, "Neg", {"sin"}},
      {{"dx"}, "Mul", {"dy", "neg"}},  // dy * (-sin(x))
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Cos", CosGrad);

Status AcosGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_21(mht_21_v, 510, "", "./tensorflow/core/ops/math_grad.cc", "AcosGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"b"}, "Sqrt", {"a"}},
    {{"inv"}, "Reciprocal", {"b"}},
    {{"neg"}, "Neg", {"inv"}},
    {{"dx"}, "Mul", {"dy", "neg"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Acos", AcosGrad);

Status AsinGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_22(mht_22_v, 529, "", "./tensorflow/core/ops/math_grad.cc", "AsinGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"b"}, "Sqrt", {"a"}},
    {{"inv"}, "Reciprocal", {"b"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Asin", AsinGrad);

Status AtanGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_23(mht_23_v, 547, "", "./tensorflow/core/ops/math_grad.cc", "AtanGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Add", {"one", "x2"}}, // 1 + x^2
    {{"inv"}, "Reciprocal", {"a"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Atan", AtanGrad);

Status TanGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_24(mht_24_v, 564, "", "./tensorflow/core/ops/math_grad.cc", "TanGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
    {{"cosx"}, "Cos", {"x"}},
    {{"secx"}, "Reciprocal", {"cosx"}},
    {{"secx2"}, "Square", {"secx"}},
    {{"dx"}, "Mul", {"dy", "secx2"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Tan", TanGrad);

Status RealGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_25(mht_25_v, 579, "", "./tensorflow/core/ops/math_grad.cc", "RealGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"dy", "zero"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Real", RealGrad);

Status ImagGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_26(mht_26_v, 592, "", "./tensorflow/core/ops/math_grad.cc", "ImagGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"zero", "dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Imag", ImagGrad);

Status AngleGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_27(mht_27_v, 605, "", "./tensorflow/core/ops/math_grad.cc", "AngleGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"re"}, "Real", {"x"}},
      {{"im"}, "Imag", {"x"}},
      {{"z"}, "Complex", {"im", "re"}},
      {{"z_inv"}, "Reciprocal", {"z"}},
      {{"neg"}, "Neg", {"z_inv"}},
      {{"dx"}, "Mul", {"neg", "dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Angle", AngleGrad);

Status ConjGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_28(mht_28_v, 622, "", "./tensorflow/core/ops/math_grad.cc", "ConjGrad");

  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Conj", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Conj", ConjGrad);

Status CastGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_29(mht_29_v, 634, "", "./tensorflow/core/ops/math_grad.cc", "CastGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: SrcT", "dy: DstT"},
      // Ret val defs
      {"dx: SrcT"},
      // Attr defs
      {{"SrcT: type"}, {"DstT: type"}},
      // Nodes
      {{{"dx"}, "Cast", {"dy"}, {{"SrcT", "$DstT"}, {"DstT", "$SrcT"}}}});
  return Status::OK();
  // clang-format on
}
REGISTER_OP_GRADIENT("Cast", CastGrad);

// Cwise binary ops
//
// TODO(zhifengc): This can be arrange as a function in the standard
// library.
Status GradForBinaryCwise(FunctionDef* g, std::vector<FDH::Node> body) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_30(mht_30_v, 657, "", "./tensorflow/core/ops/math_grad.cc", "GradForBinaryCwise");

  // clang-format off
  std::vector<FDH::Node> nodes = {
    {{"sx"}, "Shape", {"x"}},
    {{"sy"}, "Shape", {"y"}},
  };
  nodes.insert(nodes.end(), body.begin(), body.end());
  std::vector<FDH::Node> reshapes = {
    {{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "sy"}},
    {{"sum_gx"}, "Sum", {"gx", "rx"}},
    {{"dx"}, "Reshape", {"sum_gx", "sx"}},
    {{"sum_gy"}, "Sum", {"gy", "ry"}},
    {{"dy"}, "Reshape", {"sum_gy", "sy"}},
  };
  nodes.insert(nodes.end(), reshapes.begin(), reshapes.end());

  // clang-format on
  for (auto& n : nodes) {
    // "BroadcastGradientArgs" doesn't need any attrs.
    if (n.attr.empty() && n.op != "BroadcastGradientArgs") {
      n.attr = {{"T", "$T"}};
    }
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status AddGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_31(mht_31_v, 695, "", "./tensorflow/core/ops/math_grad.cc", "AddGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Identity", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Add", AddGrad);
REGISTER_OP_GRADIENT("AddV2", AddGrad);

Status SubGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_32(mht_32_v, 709, "", "./tensorflow/core/ops/math_grad.cc", "SubGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Neg", {"dz"}},          // -dz
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sub", SubGrad);

Status MulGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_33(mht_33_v, 722, "", "./tensorflow/core/ops/math_grad.cc", "MulGrad");

  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
    return GradForBinaryCwise(
        g, {
               {{"cy"}, "Conj", {"y"}, {}, {"dz"}},
               {{"gx"}, "Mul", {"dz", "cy"}},  // dz * Conj(y)
               {{"cx"}, "Conj", {"x"}, {}, {"dz"}},
               {{"gy"}, "Mul", {"cx", "dz"}},  // Conj(x) * dz
           });
  } else {
    // clang-format off
    return GradForBinaryCwise(g, {
        {{"gx"}, "Mul", {"dz", "y"}},  // dz * y
        {{"gy"}, "Mul", {"x", "dz"}},  // x * dz
    });
    // clang-format on
  }
}
REGISTER_OP_GRADIENT("Mul", MulGrad);

Status MulNoNanGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_34(mht_34_v, 747, "", "./tensorflow/core/ops/math_grad.cc", "MulNoNanGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "MulNoNan", {"y", "dz"}},  // y * dz
      {{"gy"}, "MulNoNan", {"x", "dz"}},  // x * dz
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("MulNoNan", MulGrad);

Status DivGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_35(mht_35_v, 760, "", "./tensorflow/core/ops/math_grad.cc", "DivGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Div", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "Div", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Div", DivGrad);

Status RealDivGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_36(mht_36_v, 776, "", "./tensorflow/core/ops/math_grad.cc", "RealDivGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "RealDiv", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "RealDiv", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("RealDiv", RealDivGrad);

Status DivNoNanGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_37(mht_37_v, 792, "", "./tensorflow/core/ops/math_grad.cc", "DivNoNanGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "DivNoNan", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "DivNoNan", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("DivNoNan", DivNoNanGrad);

Status PowGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_38(mht_38_v, 808, "", "./tensorflow/core/ops/math_grad.cc", "PowGrad");

  // clang-format off
  std::vector<FDH::Node> nodes = {
    {{"z"}, "Pow", {"x", "y"}},
    // dz * y * Pow(x, y - 1)
    FDH::Const("const_zero", 0.0f),
    FDH::Const("const_one", 1.0f),
    {{"zero"}, "Cast", {"const_zero"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"one"}, "Cast", {"const_one"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"t0"}, "Sub", {"y", "one"}, {}, {"dz"}},
    {{"t1"}, "Pow", {"x", "t0"}},
    {{"t2"}, "Mul", {"dz", "y"}},
    {{"gx"}, "Mul", {"t1", "t2"}},
    {{"unsafe_log"}, "Log", {"x"}, {}, {"dz"}},
    {{"zeros"}, "ZerosLike", {"x"}}};
  // clang-format on
  std::vector<FDH::Node> log_x_handling;
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
    // dz * z * (x != 0 ? Log(x) : 0)
    // clang-format off
    log_x_handling = {
      {{"nz_x"}, "NotEqual", {"x", "zero"}},
      {{"safe_log"}, "Select", {"nz_x", "unsafe_log", "zeros"}}};
    // clang-format on
  } else {
    // dz * z * (x > 0 ? Log(x) : 0)
    // clang-format off
    log_x_handling = {
      {{"pos_x"}, "Greater", {"x", "zero"}},
      {{"safe_log"}, "Select", {"pos_x", "unsafe_log", "zeros"}}};
    // clang-format on
  }
  nodes.insert(nodes.end(), log_x_handling.begin(), log_x_handling.end());
  nodes.push_back({{"t4"}, "Mul", {"dz", "z"}});
  nodes.push_back({{"gy"}, "Mul", {"safe_log", "t4"}});
  return GradForBinaryCwise(g, nodes);
}
REGISTER_OP_GRADIENT("Pow", PowGrad);

Status XlogyGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_39(mht_39_v, 852, "", "./tensorflow/core/ops/math_grad.cc", "XlogyGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_logy"}, "Xlogy", {"is_zero_cast", "y"}},
      {{"xlogygrad"}, "Xdivy", {"x", "y"}},
      {{"gx"}, "Mul", {"safe_logy", "dz"}},
      {{"gy"}, "Mul", {"xlogygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xlogy", XlogyGrad);

Status Xlog1pyGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_40(mht_40_v, 871, "", "./tensorflow/core/ops/math_grad.cc", "Xlog1pyGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"yp1"}, "Add", {"y", "one"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_log1py"}, "Xlog1py", {"is_zero_cast", "y"}},
      {{"xlog1pygrad"}, "Xdivy", {"x", "yp1"}},
      {{"gx"}, "Mul", {"safe_log1py", "dz"}},
      {{"gy"}, "Mul", {"xlog1pygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xlog1py", Xlog1pyGrad);

Status XdivyGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_41(mht_41_v, 893, "", "./tensorflow/core/ops/math_grad.cc", "XdivyGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_divy"}, "Xdivy", {"is_zero_cast", "y"}},
      {{"y2"}, "Square", {"y"}},
      {{"negy2"}, "Neg", {"y2"}},
      {{"xdivygrad"}, "Xdivy", {"x", "negy2"}},
      {{"gx"}, "Mul", {"safe_divy", "dz"}},
      {{"gy"}, "Mul", {"xdivygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xdivy", XdivyGrad);

Status SquaredDifferenceGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_42(mht_42_v, 914, "", "./tensorflow/core/ops/math_grad.cc", "SquaredDifferenceGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      FDH::Const("c", int64_t{2}),
      {{"two"}, "Cast", {"c"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
      {{"x_sub_y"}, "Sub", {"x", "y"}},
      {{"two_x_sub_y"}, "Mul", {"two", "x_sub_y"}},  // 2 * (x - y)
      {{"gx"}, "Mul", {"two_x_sub_y", "dz"}},
      {{"gy"}, "Neg", {"gx"}}
    });
  // clang-format on
}
REGISTER_OP_GRADIENT("SquaredDifference", SquaredDifferenceGrad);

Status MaximumMinimumGradHelper(const string& comparator,
                                const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("comparator: \"" + comparator + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_43(mht_43_v, 933, "", "./tensorflow/core/ops/math_grad.cc", "MaximumMinimumGradHelper");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"c"}, comparator, {"x", "y"}, {}, {"dz"}},
      {{"mask"}, "Cast", {"c"}, {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"gx"}, "Mul", {"dz", "mask"}},
      {{"gy"}, "Sub", {"dz", "gx"}},
  });
  // clang-format on
}

Status MaximumGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_44(mht_44_v, 947, "", "./tensorflow/core/ops/math_grad.cc", "MaximumGrad");

  return MaximumMinimumGradHelper("GreaterEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Maximum", MaximumGrad);

Status MinimumGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_45(mht_45_v, 955, "", "./tensorflow/core/ops/math_grad.cc", "MinimumGrad");

  return MaximumMinimumGradHelper("LessEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Minimum", MinimumGrad);

Status ComplexGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_46(mht_46_v, 963, "", "./tensorflow/core/ops/math_grad.cc", "ComplexGrad");

  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Real", {"dz"}},
      {{"gy"}, "Imag", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Complex", ComplexGrad);

// Cwise ternary ops.
Status SelectGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_47(mht_47_v, 977, "", "./tensorflow/core/ops/math_grad.cc", "SelectGrad");

  // clang-format off
  *g = FDH::Define(
      {"c:bool", "x:T", "y:T", "dz:T"},
      {"dc:bool", "dx:T", "dy:T"},
      {{"T: {half, float, double}"}},
      {
        {{"dc"}, "ZerosLike", {"c"}, {{"T", DT_BOOL}}, {"dz"}},
        {{"zeros"}, "ZerosLike", {"x"}, {{"T", "$T"}}, {"dz"}},
        {{"dx"}, "Select", {"c", "dz", "zeros"}, {{"T", "$T"}}},
        {{"dy"}, "Select", {"c", "zeros", "dz"}, {{"T", "$T"}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Select", SelectGrad);

// N-ry ops
// REGISTER_OP_GRADIENT("AddN", AddNGrad);

// Reduction ops
//
// TODO(zhifengc): This helper is pretty ugly. Do something better.
// TODO(zhifengc): This can be arrange as a function in the standard library.
Status GradForReductionOp(FunctionDef* g, std::vector<FDH::Node> body) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_48(mht_48_v, 1004, "", "./tensorflow/core/ops/math_grad.cc", "GradForReductionOp");

  // Shape manipulation nodes.

  // clang-format off
  std::vector<FDH::Node> nodes = {
   {{"x_shape"}, "Shape", {"x"}},
   {{"x_rank"}, "Rank", {"x"}},
   {{"i_shape"}, "Shape", {"i"}, {{"T", DT_INT32}}},
   FDH::Const("zero", 0),
   FDH::Const("one", 1),
   // stitch_idx0 = Range(0, x_rank, 1)
   {{"stitch_val1"}, "Fill", {"i_shape:output:0", "one:output:0"},
    {{"T", DT_INT32}}},
   {{"y_shape"}, "DynamicStitch",
    {"stitch_idx0:output:0", "i",
     "x_shape:output:0", "stitch_val1:output:0"},
    {{"N", 2}, {"T", DT_INT32}}},
   {{"tile_scaling"}, "Div", {"x_shape:output:0", "y_shape:merged:0"},
    {{"T", DT_INT32}}},
   {{"di"}, "ZerosLike", {"i"}, {{"T", DT_INT32}}}
  };
  // clang-format on
  nodes.insert(nodes.end(), body.begin(), body.end());
  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", "$T"}};
    }
  }
  // "Range" doesn't need any attr.
  nodes.push_back({{"stitch_idx0"},
                   "Range",
                   {"zero:output:0", "x_rank:output:0", "one:output:0"},
                   {}});
  *g = FDH::Create("_",
                   // Input defs
                   {"x:T", "i:int32", "dy:T"},
                   // Ret val defs
                   {"dx:T", "di:int32"},
                   // Attr defs
                   {{"T: {half, float, double}"}},
                   // Nodes
                   nodes,
                   // Return values
                   {{"dx", "dx:output:0"}, {"di", "di:y:0"}});
  return Status::OK();
}

Status SumGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_49(mht_49_v, 1054, "", "./tensorflow/core/ops/math_grad.cc", "SumGrad");

  // clang-format off
  return GradForReductionOp(g, {
    {{"dy_reshaped"}, "Reshape", {"dy", "y_shape:merged:0"}},
    {{"dx"}, "Tile", {"dy_reshaped:output:0", "tile_scaling:z:0"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sum", SumGrad);

Status MeanGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_50(mht_50_v, 1067, "", "./tensorflow/core/ops/math_grad.cc", "MeanGrad");

  // clang-format off
  return GradForReductionOp(g, {
    {{"factor"}, "Prod", {"tile_scaling:z:0", "zero:output:0"},
                   {{"T", DT_INT32}}},
    {{"factor_T"}, "Cast", {"factor:output:0"},
                   {{"SrcT", DT_INT32}, {"DstT", "$T"}}},
    {{"dy_scaled"}, "Div", {"dy", "factor_T:y:0"}},
    {{"dy_reshaped"}, "Reshape", {"dy_scaled:z:0", "y_shape:merged:0"}},
    {{"dx"}, "Tile", {"dy_reshaped:output:0", "tile_scaling:z:0"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Mean", MeanGrad);

// REGISTER_OP_GRADIENT("Prod", ProdGrad);
// REGISTER_OP_GRADIENT("SegmentSum", SegmentSumGrad);
// REGISTER_OP_GRADIENT("SegmentMean", SegmentMeanGrad);
// REGISTER_OP_GRADIENT("SparseSegmentSum", SparseSegmentSumGrad);
// REGISTER_OP_GRADIENT("SparseSegmentMean", SparseSegmentMeanGrad);
// REGISTER_OP_GRADIENT("SparseSegmentSqrtN", SparseSegmentSqrtNGrad);
// REGISTER_OP_GRADIENT("SegmentMin", SegmentMinGrad);
// REGISTER_OP_GRADIENT("SegmentMax", SegmentMaxGrad);
// REGISTER_OP_GRADIENT("UnsortedSegmentSum", UnsortedSegmentSumGrad);
// REGISTER_OP_GRADIENT("UnsortedSegmentMax", UnsortedSegmentMaxGrad);

Status MinMaxGradHelper(const string& op, const AttrSlice& attrs,
                        FunctionDef* g) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_51(mht_51_v, 1098, "", "./tensorflow/core/ops/math_grad.cc", "MinMaxGradHelper");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x:T", "i:int32", "dy:T"},
      // Ret val defs
      {"dx:T", "di:int32"},
      // Attr defs
      {{"T: {half, float, double}"}},
      {
        // keep_dims because we need to do x == y, which requires x
        // and y are broadcastable.
        {{"y"}, op, {"x", "i"}, {{"T", "$T"}, {"keep_dims", true}}},
        {{"mask"}, "Equal", {"x", "y"}, {{"T", "$T"}}},
        {{"mask_cast"}, "Cast", {"mask"}, {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
        {{"mask_sum"}, "Sum", {"mask_cast", "i"}, {{"T", "$T"}}},
        {{"norm_dy"}, "Div", {"dy", "mask_sum"}, {{"T", "$T"}}},
        {{"sy"}, "Shape", {"y"}, {{"T", "$T"}}},
        {{"norm_dy_reshaped"}, "Reshape", {"norm_dy", "sy"}, {{"T", "$T"}}},
        {{"dx"}, "Mul", {"mask_cast", "norm_dy_reshaped"}, {{"T", "$T"}}},
        {{"di"}, "ZerosLike", {"i"}, {{"T", DT_INT32}}}
      });
  // clang-format on
  return Status::OK();
}

Status MaxGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_52(mht_52_v, 1127, "", "./tensorflow/core/ops/math_grad.cc", "MaxGrad");

  return MinMaxGradHelper("Max", attrs, g);
}
REGISTER_OP_GRADIENT("Max", MaxGrad);

Status MinGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_53(mht_53_v, 1135, "", "./tensorflow/core/ops/math_grad.cc", "MinGrad");

  return MinMaxGradHelper("Min", attrs, g);
}
REGISTER_OP_GRADIENT("Min", MinGrad);

static Status MatMulGradHelper(FunctionDef* g, const string& opname,
                               const string& attr_adj_x,
                               const string& attr_adj_y, const string& x0,
                               bool ax0, const string& x1, bool ax1,
                               const string& y0, bool ay0, const string& y1,
                               bool ay1, bool enable_broadcasting) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("opname: \"" + opname + "\"");
   mht_54_v.push_back("attr_adj_x: \"" + attr_adj_x + "\"");
   mht_54_v.push_back("attr_adj_y: \"" + attr_adj_y + "\"");
   mht_54_v.push_back("x0: \"" + x0 + "\"");
   mht_54_v.push_back("x1: \"" + x1 + "\"");
   mht_54_v.push_back("y0: \"" + y0 + "\"");
   mht_54_v.push_back("y1: \"" + y1 + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_54(mht_54_v, 1155, "", "./tensorflow/core/ops/math_grad.cc", "MatMulGradHelper");

  // The final outputs are "dx" and "dy". If we're broadcasting compute
  // intermediate nodes for now.
  std::vector<FDH::Node> nodes = {
      {{(enable_broadcasting ? "gx" : "dx")},
       opname,
       {x0, x1},
       {{"T", "$T"}, {attr_adj_x, ax0}, {attr_adj_y, ax1}}},
      {{(enable_broadcasting ? "gy" : "dy")},
       opname,
       {y0, y1},
       {{"T", "$T"}, {attr_adj_x, ay0}, {attr_adj_y, ay1}}},
  };
  // TODO(anudhyan): Figure out a way to inspect the static shapes of "x" and
  // "y". If they have the same batch dimensions, then we can omit adding the
  // broadcasting-specific ops.
  if (enable_broadcasting) {
    std::vector<FDH::Node> unbroadcast_gradients = {
        FDH::Const<int32>("zero", gtl::ArraySlice<int32>{0}),
        FDH::Const<int32>("one", gtl::ArraySlice<int32>{1}),
        FDH::Const<int32>("minustwo", gtl::ArraySlice<int32>{-2}),
        // Compute the batch shapes of the inputs (all but last two dims).
        {{"sx"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"sy"}, "Shape", {"y"}, {{"T", "$T"}}},
        {{"batch_sx"},
         "StridedSlice",
         {"sx", "zero", "minustwo", "one"},
         {{"T", DT_INT32}, {"Index", DT_INT32}}},
        {{"batch_sy"},
         "StridedSlice",
         {"sy", "zero", "minustwo", "one"},
         {{"T", DT_INT32}, {"Index", DT_INT32}}},
        // Sum along dimensions that the inputs were broadcasted across.
        {{"rx", "ry"}, "BroadcastGradientArgs", {"batch_sx", "batch_sy"}},
        {{"sum_gx"}, "Sum", {"gx", "rx"}, {{"T", "$T"}}},
        {{"sum_gy"}, "Sum", {"gy", "ry"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"sum_gx", "sx"}, {{"T", "$T"}}},
        {{"dy"}, "Reshape", {"sum_gy", "sy"}, {{"T", "$T"}}}};
    nodes.insert(nodes.end(), unbroadcast_gradients.begin(),
                 unbroadcast_gradients.end());
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status MatMulGradCommon(const string& opname, const string& attr_adj_x,
                        const string& attr_adj_y, const AttrSlice& attrs,
                        FunctionDef* g, bool enable_broadcasting) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("opname: \"" + opname + "\"");
   mht_55_v.push_back("attr_adj_x: \"" + attr_adj_x + "\"");
   mht_55_v.push_back("attr_adj_y: \"" + attr_adj_y + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_55(mht_55_v, 1216, "", "./tensorflow/core/ops/math_grad.cc", "MatMulGradCommon");

  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
    return errors::Unimplemented(
        "MatMul gradient for complex is not supported yet.");
  }
  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_x, &ta));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_y, &tb));
  if (!ta && !tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                            true, "x", true, "dz", false, enable_broadcasting);
  }
  if (!ta && tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                            false, "dz", true, "x", false, enable_broadcasting);
  }
  if (ta && !tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", false, "dz",
                            true, "x", false, "dz", false, enable_broadcasting);
  }
  CHECK(ta && tb);
  return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", true, "dz",
                          true, "dz", true, "x", true, enable_broadcasting);
}

Status MatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_56(mht_56_v, 1247, "", "./tensorflow/core/ops/math_grad.cc", "MatMulGrad");

  return MatMulGradCommon("MatMul", "transpose_a", "transpose_b", attrs, g,
                          false /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("MatMul", MatMulGrad);

Status BatchMatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_57(mht_57_v, 1256, "", "./tensorflow/core/ops/math_grad.cc", "BatchMatMulGrad");

  return MatMulGradCommon("BatchMatMul", "adj_x", "adj_y", attrs, g,
                          false /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("BatchMatMul", BatchMatMulGrad);

Status BatchMatMulV2Grad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_gradDTcc mht_58(mht_58_v, 1265, "", "./tensorflow/core/ops/math_grad.cc", "BatchMatMulV2Grad");

  return MatMulGradCommon("BatchMatMulV2", "adj_x", "adj_y", attrs, g,
                          true /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("BatchMatMulV2", BatchMatMulV2Grad);

// REGISTER_OP_GRADIENT("SparseMatMul", SparseMatMulGrad);

// Comparison ops.
REGISTER_OP_NO_GRADIENT("Less");
REGISTER_OP_NO_GRADIENT("LessEqual");
REGISTER_OP_NO_GRADIENT("Greater");
REGISTER_OP_NO_GRADIENT("GreaterEqual");
REGISTER_OP_NO_GRADIENT("Equal");
REGISTER_OP_NO_GRADIENT("NotEqual");

// Logical ops.
REGISTER_OP_NO_GRADIENT("LogicalAnd");
REGISTER_OP_NO_GRADIENT("LogicalOr");
REGISTER_OP_NO_GRADIENT("LogicalNot");

// Sequence generation ops.
REGISTER_OP_NO_GRADIENT("Range");
REGISTER_OP_NO_GRADIENT("LinSpace");

REGISTER_OP_NO_GRADIENT("Floor");
REGISTER_OP_NO_GRADIENT("FloorDiv");
REGISTER_OP_NO_GRADIENT("TruncateDiv");

}  // end namespace tensorflow
