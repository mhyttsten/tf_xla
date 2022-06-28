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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

XlaComputation CreateScalarComputation(const std::string& name,
                                       PrimitiveType type, XlaBuilder* builder,
                                       XlaOpGenerator generator) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarComputation");

  std::unique_ptr<XlaBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        absl::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = Parameter(b.get(), 0, scalar, "lhs");
  auto rhs = Parameter(b.get(), 1, scalar, "rhs");
  generator(lhs, rhs);
  return b->BuildAndNoteError();
}

XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarAddComputation");

  return CreateScalarComputation(
      "add", type, builder, [](XlaOp lhs, XlaOp rhs) { return Add(lhs, rhs); });
}

XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarMultiplyComputation");

  return CreateScalarComputation(
      "mul", type, builder, [](XlaOp lhs, XlaOp rhs) { return Mul(lhs, rhs); });
}

XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_3(mht_3_v, 241, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarGeComputation");

  return CreateScalarComputation(
      "ge", type, builder, [](XlaOp lhs, XlaOp rhs) { return Ge(lhs, rhs); });
}

XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarMaxComputation");

  return CreateScalarComputation(
      "max", type, builder, [](XlaOp lhs, XlaOp rhs) { return Max(lhs, rhs); });
}

XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_5(mht_5_v, 259, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarMinComputation");

  return CreateScalarComputation(
      "min", type, builder, [](XlaOp lhs, XlaOp rhs) { return Min(lhs, rhs); });
}

XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_6(mht_6_v, 268, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarAndComputation");

  return CreateScalarComputation(
      "and", type, builder, [](XlaOp lhs, XlaOp rhs) { return And(lhs, rhs); });
}

XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_7(mht_7_v, 277, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarOrComputation");

  return CreateScalarComputation(
      "or", type, builder, [](XlaOp lhs, XlaOp rhs) { return Or(lhs, rhs); });
}

XlaComputation CreateScalarIdentityWithZeroComputation(PrimitiveType type,
                                                       XlaBuilder* builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_8(mht_8_v, 286, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateScalarIdentityWithZeroComputation");

  XlaComputation reducer =
      (primitive_util::IsIntegralType(type) || type == PRED)
          ? CreateScalarOrComputation(type, builder)
          : CreateScalarAddComputation(type, builder);
  return reducer;
}

XlaOp Any(XlaOp predicates) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_9(mht_9_v, 297, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "Any");

  XlaBuilder* builder = predicates.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto f = ConstantR0<bool>(builder, false);
    XlaComputation logical_or = CreateScalarOrComputation(PRED, builder);
    TF_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                        builder->GetShape(predicates));
    std::vector<int64_t> all_dimensions(predicates_shape.rank());
    std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
    return Reduce(predicates, f, logical_or, all_dimensions);
  });
}

static XlaComputation CreateMinMaxComputation(XlaBuilder* outer_builder,
                                              PrimitiveType value_type,
                                              PrimitiveType index_type,
                                              bool is_min) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_10(mht_10_v, 316, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "CreateMinMaxComputation");

  auto sub_builder = outer_builder->CreateSubBuilder("minmax_func");
  XlaBuilder* b = sub_builder.get();
  XlaOp lhs_value =
      Parameter(b, 0, ShapeUtil::MakeShape(value_type, {}), "lhs_value");
  XlaOp lhs_index =
      Parameter(b, 1, ShapeUtil::MakeShape(index_type, {}), "lhs_index");
  XlaOp rhs_value =
      Parameter(b, 2, ShapeUtil::MakeShape(value_type, {}), "rhs_value");
  XlaOp rhs_index =
      Parameter(b, 3, ShapeUtil::MakeShape(index_type, {}), "rhs_index");

  XlaOp cmp = is_min ? Le(lhs_value, rhs_value) : Ge(lhs_value, rhs_value);
  XlaOp max = Select(cmp, lhs_value, rhs_value);
  XlaOp arg_max = Select(cmp, lhs_index, rhs_index);
  XlaOp eq = Eq(lhs_value, rhs_value);
  XlaOp tie_id = Min(lhs_index, rhs_index);
  arg_max = Select(eq, tie_id, arg_max);
  Tuple(b, {max, arg_max});
  return b->BuildAndNoteError();
}

XlaOp ArgMinMax(XlaOp input, PrimitiveType output_type, int axis, bool is_min) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_11(mht_11_v, 341, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "ArgMinMax");

  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    XlaOp value_init_value;
    if (is_min) {
      value_init_value = MaxValue(builder, input_shape.element_type());
    } else {
      value_init_value = MinValue(builder, input_shape.element_type());
    }
    int64_t dimension_size = input_shape.dimensions(axis);
    auto index_type = dimension_size <= INT32_MAX ? S32 : output_type;
    XlaOp index_init_value = Zero(builder, index_type);
    auto iota_shape = input_shape;
    iota_shape.set_element_type(index_type);
    XlaOp iota = Iota(builder, iota_shape, axis);

    XlaComputation reducer = CreateMinMaxComputation(
        builder, input_shape.element_type(), index_type, is_min);
    XlaOp max_argmax = Reduce(builder, {input, iota},
                              {value_init_value, index_init_value}, reducer,
                              /*dimensions_to_reduce=*/{axis});
    XlaOp argmax = GetTupleElement(max_argmax, 1);
    if (index_type != output_type) {
      argmax = ConvertElementType(argmax, output_type);
    }
    return argmax;
  });
}

XlaOp ArgMax(XlaOp input, PrimitiveType output_type, int axis) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_12(mht_12_v, 374, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "ArgMax");

  return ArgMinMax(input, output_type, axis, /*is_min=*/false);
}

XlaOp ArgMin(XlaOp input, PrimitiveType output_type, int axis) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSarithmeticDTcc mht_13(mht_13_v, 381, "", "./tensorflow/compiler/xla/client/lib/arithmetic.cc", "ArgMin");

  return ArgMinMax(input, output_type, axis, /*is_min=*/true);
}

}  // namespace xla
