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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/comparators.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using XlaCompareOp = XlaOp (*)(XlaOp, XlaOp, absl::Span<const int64_t>);

XlaComputation CreateScalarComparisonComputation(
    const std::string& name, const std::vector<PrimitiveType>& operand_types,
    XlaBuilder* builder, XlaCompareOp generator) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/client/lib/comparators.cc", "CreateScalarComparisonComputation");

  CHECK_NE(operand_types.size(), 0);
  std::vector<absl::optional<XlaCompareOp>> generators(operand_types.size());
  generators[0] = generator;
  return CreateScalarComparisonComputation(name, operand_types, generators,
                                           builder);
}
}  // namespace

XlaComputation CreateScalarComparisonComputation(
    const std::string& name, const std::vector<PrimitiveType>& operand_types,
    const std::vector<absl::optional<XlaCompareOp>>& generators,
    XlaBuilder* builder) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/xla/client/lib/comparators.cc", "CreateScalarComparisonComputation");

  // Create a default computation where we compare only the first two
  // parameters of type 'operand_types[0]'.
  auto b = builder->CreateSubBuilder(name);
  if (operand_types.empty()) {
    b->ReportError(InvalidArgument("operand_types should not be empty"));
    return b->BuildAndNoteError();
  }

  CHECK_EQ(operand_types.size(), generators.size());
  int64_t parameter_count = 0;
  int64_t last_generator_index = 0;
  std::vector<XlaOp> lhs_params;
  std::vector<XlaOp> rhs_params;

  // For each type in 'operand_types' we create two parameters of this type. The
  // idea is that this computation can be used by n-ary Sort, and potentially
  // should support comparing also the other operands of sort. In this default
  // computation, however, we will not actually use any parameters except the
  // first two.
  for (auto operand_type : operand_types) {
    auto scalar_shape = ShapeUtil::MakeShape(operand_type, {});
    auto lhs_param = Parameter(b.get(), parameter_count * 2, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".lhs"));
    auto rhs_param = Parameter(b.get(), parameter_count * 2 + 1, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".rhs"));
    lhs_params.emplace_back(lhs_param);
    rhs_params.emplace_back(rhs_param);
    if (generators[parameter_count].has_value()) {
      last_generator_index = parameter_count;
    }
    parameter_count++;
  }

  CHECK_NE(parameter_count, 0);

  auto shape_or = b->GetShape(lhs_params[0]);
  if (!shape_or.ok()) {
    b->ReportError(shape_or.status());
    return {};
  }
  Shape shape = shape_or.ValueOrDie();
  shape.set_element_type(PRED);
  XlaOp param_equal =
      Broadcast(One(b.get(), shape.element_type()), shape.dimensions());
  XlaOp result = param_equal;

  for (int64_t i = 0; i < parameter_count; i++) {
    if (generators[i].has_value()) {
      result = Select(param_equal,
                      generators[i].value()(lhs_params[i], rhs_params[i], {}),
                      result);
      if (i != last_generator_index) {
        param_equal =
            And(param_equal, EqTotalOrder(lhs_params[i], rhs_params[i]));
      }
    }
  }

  return b->BuildAndNoteError();
}

// Creates a scalar less-than computation and returns it.
XlaComputation CreateScalarLtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc mht_2(mht_2_v, 292, "", "./tensorflow/compiler/xla/client/lib/comparators.cc", "CreateScalarLtComputation");

  return CreateScalarComparisonComputation("compare-less-than", operand_types,
                                           builder, LtTotalOrder);
}

// Creates a scalar greater-than computation and returns it.
XlaComputation CreateScalarGtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPScomparatorsDTcc mht_3(mht_3_v, 302, "", "./tensorflow/compiler/xla/client/lib/comparators.cc", "CreateScalarGtComputation");

  return CreateScalarComparisonComputation(
      "compare-greater-than", operand_types, builder, GtTotalOrder);
}

}  // namespace xla
