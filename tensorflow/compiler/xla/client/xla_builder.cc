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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc() {
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

#include "tensorflow/compiler/xla/client/xla_builder.h"

#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

using absl::StrCat;

namespace {

static const char kNameSeparator = '.';

// Retrieves the base name of an instruction or computation fully qualified
// name, using separator as boundary between the initial base name part, and
// the numeric identification.
std::string GetBaseName(const std::string& name, char separator) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("separator: '" + std::string(1, separator) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_0(mht_0_v, 236, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GetBaseName");

  auto pos = name.rfind(separator);
  CHECK_NE(pos, std::string::npos) << name;
  return name.substr(0, pos);
}

// Generates a fully qualified computation/instruction name.
std::string GetFullName(const std::string& base_name, char separator,
                        int64_t id) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("base_name: \"" + base_name + "\"");
   mht_1_v.push_back("separator: '" + std::string(1, separator) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GetFullName");

  const char separator_str[] = {separator, '\0'};
  return StrCat(base_name, separator_str, id);
}

// Common function to standardize setting name and IDs on computation and
// instruction proto entities.
template <typename T>
void SetProtoIdAndName(T* entry, const std::string& base_name, char separator,
                       int64_t id) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("base_name: \"" + base_name + "\"");
   mht_2_v.push_back("separator: '" + std::string(1, separator) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SetProtoIdAndName");

  entry->set_id(id);
  entry->set_name(GetFullName(base_name, separator, id));
}

bool InstrIsSetBound(const HloInstructionProto* instr_proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "InstrIsSetBound");

  HloOpcode opcode = StringToHloOpcode(instr_proto->opcode()).ValueOrDie();
  if (opcode == HloOpcode::kCustomCall &&
      instr_proto->custom_call_target() == "SetBound") {
    return true;
  }
  return false;
}

}  // namespace

namespace internal {

XlaOp XlaBuilderFriend::BuildFusion(XlaBuilder* builder,
                                    absl::Span<const XlaOp> operands,
                                    absl::string_view fusion_kind,
                                    const XlaComputation& fused_computation) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fusion_kind: \"" + std::string(fusion_kind.data(), fusion_kind.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_4(mht_4_v, 291, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilderFriend::BuildFusion");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    instr.set_fusion_kind(std::string(fusion_kind));
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(auto program_shape,
                        fused_computation.GetProgramShape());
    *instr.mutable_shape() = program_shape.result().ToProto();
    builder->AddCalledComputation(fused_computation, &instr);
    return builder->AddInstruction(std::move(instr), HloOpcode::kFusion,
                                   operands);
  });
}

XlaOp XlaBuilderFriend::BuildBitcast(XlaBuilder* builder, XlaOp operand,
                                     const Shape& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_5(mht_5_v, 309, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilderFriend::BuildBitcast");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return builder->AddInstruction(std::move(instr), HloOpcode::kBitcast,
                                   {operand});
  });
}

XlaOp XlaBuilderFriend::BuildRngGetAndUpdateState(XlaBuilder* builder,

                                                  int64_t delta,
                                                  const Shape& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_6(mht_6_v, 324, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilderFriend::BuildRngGetAndUpdateState");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    instr.set_delta(delta);
    *instr.mutable_shape() = shape.ToProto();
    return builder->AddInstruction(std::move(instr),
                                   HloOpcode::kRngGetAndUpdateState);
  });
}

HloInstructionProto* XlaBuilderFriend::GetInstruction(XlaOp op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_7(mht_7_v, 337, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilderFriend::GetInstruction");

  return &op.builder()
              ->instructions_[op.builder()->handle_to_index_[op.handle_]];
}

HloInstructionProto* XlaBuilderFriend::GetInstructionByHandle(
    XlaBuilder* builder, int64_t handle) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_8(mht_8_v, 346, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilderFriend::GetInstructionByHandle");

  return &builder->instructions_[builder->handle_to_index_[handle]];
}

}  // namespace internal

XlaOp operator-(XlaOp x) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_9(mht_9_v, 355, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "-");
 return Neg(x); }
XlaOp operator+(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_10(mht_10_v, 359, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "+");
 return Add(x, y); }
XlaOp operator-(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_11(mht_11_v, 363, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "-");
 return Sub(x, y); }
XlaOp operator*(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_12(mht_12_v, 367, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "*");
 return Mul(x, y); }
XlaOp operator/(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_13(mht_13_v, 371, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "/");
 return Div(x, y); }
XlaOp operator%(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_14(mht_14_v, 375, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "%");
 return Rem(x, y); }

XlaOp operator~(XlaOp x) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_15(mht_15_v, 380, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "operator~");
 return Not(x); }
XlaOp operator&(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_16(mht_16_v, 384, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "&");
 return And(x, y); }
XlaOp operator|(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_17(mht_17_v, 388, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "|");
 return Or(x, y); }
XlaOp operator^(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_18(mht_18_v, 392, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "^");
 return Xor(x, y); }
XlaOp operator<<(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_19(mht_19_v, 396, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "operator<<");
 return ShiftLeft(x, y); }

XlaOp operator>>(XlaOp x, XlaOp y) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_20(mht_20_v, 401, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "operator>>");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const xla::Shape* shape, builder->GetShapePtr(x));
    if (!ShapeUtil::ElementIsIntegral(*shape)) {
      return InvalidArgument(
          "Argument to >> operator does not have an integral type (%s).",
          ShapeUtil::HumanString(*shape));
    }
    if (ShapeUtil::ElementIsSigned(*shape)) {
      return ShiftRightArithmetic(x, y);
    } else {
      return ShiftRightLogical(x, y);
    }
  });
}

StatusOr<const Shape*> XlaBuilder::GetShapePtr(XlaOp op) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_21(mht_21_v, 421, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetShapePtr");

  TF_RETURN_IF_ERROR(first_error_);
  TF_RETURN_IF_ERROR(CheckOpBuilder(op));
  auto it = handle_to_index_.find(op.handle());
  if (it == handle_to_index_.end()) {
    return InvalidArgument("No XlaOp with handle %d", op.handle());
  }
  return instruction_shapes_.at(it->second).get();
}

StatusOr<Shape> XlaBuilder::GetShape(XlaOp op) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_22(mht_22_v, 434, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetShape");

  TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(op));
  return *shape;
}

StatusOr<std::vector<Shape>> XlaBuilder::GetOperandShapes(
    absl::Span<const XlaOp> operands) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_23(mht_23_v, 443, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetOperandShapes");

  std::vector<Shape> operand_shapes;
  operand_shapes.reserve(operands.size());
  for (XlaOp operand : operands) {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    operand_shapes.push_back(*shape);
  }
  return operand_shapes;
}

std::string XlaBuilder::OpToString(XlaOp op) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_24(mht_24_v, 456, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::OpToString");

  std::string s;
  ToStringHelper(&s, /*ident=*/0, op.handle());
  return s;
}

static std::string ShapeToString(const xla::ShapeProto& shape) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_25(mht_25_v, 465, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ShapeToString");

  if (shape.tuple_shapes_size() > 1) {
    return absl::StrCat(
        "(",
        absl::StrJoin(shape.tuple_shapes(), ", ",
                      [&](std::string* s, const xla::ShapeProto& subshape) {
                        absl::StrAppend(s, ShapeToString(subshape));
                      }),
        ")");
  }
  return absl::StrCat("[", absl::StrJoin(shape.dimensions(), ", "), "]");
}

void XlaBuilder::ToStringHelper(std::string* out, int ident,
                                int64_t op_handle) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_26(mht_26_v, 482, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ToStringHelper");

  const HloInstructionProto& instr =
      *(LookUpInstructionByHandle(op_handle).ValueOrDie());
  absl::StrAppend(out, std::string(ident, ' '), instr.opcode(),
                  ", shape=", ShapeToString(instr.shape()));
  if (instr.has_metadata()) {
    absl::StrAppend(out, ", metadata={", instr.metadata().source_file(), ":",
                    instr.metadata().source_line(), "}");
  }
  if (instr.operand_ids_size()) {
    absl::StrAppend(out, "\n");
  }
  absl::StrAppend(out, absl::StrJoin(instr.operand_ids(), "\n",
                                     [&](std::string* s, int64_t subop) {
                                       ToStringHelper(s, ident + 2, subop);
                                     }));
}

XlaBuilder::XlaBuilder(const std::string& computation_name)
    : name_(computation_name) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("computation_name: \"" + computation_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_27(mht_27_v, 505, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::XlaBuilder");
}

XlaBuilder::~XlaBuilder() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_28(mht_28_v, 510, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::~XlaBuilder");
}

XlaOp XlaBuilder::ReportError(const Status& error) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_29(mht_29_v, 515, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReportError");

  CHECK(!error.ok());
  if (die_immediately_on_error_) {
    LOG(FATAL) << "error building computation: " << error;
  }

  if (first_error_.ok()) {
    first_error_ = error;
    first_error_backtrace_.CreateCurrent(/*skip_count=*/1);
  }
  return XlaOp(this);
}

XlaOp XlaBuilder::ReportErrorOrReturn(const StatusOr<XlaOp>& op) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_30(mht_30_v, 531, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReportErrorOrReturn");

  if (!first_error_.ok()) {
    return XlaOp(this);
  }
  if (!op.ok()) {
    return ReportError(op.status());
  }
  return op.ValueOrDie();
}

XlaOp XlaBuilder::ReportErrorOrReturn(
    const std::function<StatusOr<XlaOp>()>& op_creator) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_31(mht_31_v, 545, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReportErrorOrReturn");

  return ReportErrorOrReturn(op_creator());
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(int64_t root_id) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_32(mht_32_v, 552, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetProgramShape");

  TF_RETURN_IF_ERROR(first_error_);
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root_proto,
                      LookUpInstructionByHandle(root_id));

  ProgramShape program_shape;

  *program_shape.mutable_result() = Shape(root_proto->shape());

  // Check that the parameter numbers are continuous from 0, and add parameter
  // shapes and names to the program shape.
  const int64_t param_count = parameter_numbers_.size();
  for (int64_t i = 0; i < param_count; i++) {
    program_shape.add_parameters();
    program_shape.add_parameter_names();
  }
  for (const HloInstructionProto& instr : instructions_) {
    // Parameter number uniqueness is guaranteed in XlaBuilder::Parameter(). So
    // to verify continuity, we just need to verify that every parameter is in
    // the right range.
    if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      const int64_t index = instr.parameter_number();
      TF_RET_CHECK(index >= 0 && index < param_count)
          << "invalid parameter number: " << index;
      *program_shape.mutable_parameters(index) = Shape(instr.shape());
      *program_shape.mutable_parameter_names(index) = instr.name();
    }
  }
  return program_shape;
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_33(mht_33_v, 586, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetProgramShape");

  TF_RET_CHECK(!instructions_.empty());
  return GetProgramShape(instructions_.back().id());
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(XlaOp root) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_34(mht_34_v, 594, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetProgramShape");

  if (root.builder_ != this) {
    return InvalidArgument("Given root operation is not in this computation.");
  }
  return GetProgramShape(root.handle());
}

void XlaBuilder::IsConstantVisitor(const int64_t op_handle, int depth,
                                   absl::flat_hash_set<int64_t>* visited,
                                   bool* is_constant) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_35(mht_35_v, 606, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::IsConstantVisitor");

  if (visited->contains(op_handle) || !*is_constant) {
    return;
  }

  const HloInstructionProto& instr =
      *(LookUpInstructionByHandle(op_handle).ValueOrDie());
  HloInstructionProto to_print(instr);
  to_print.clear_shape();
  const HloOpcode opcode = StringToHloOpcode(instr.opcode()).ValueOrDie();
  const std::string indent =
      absl::StrJoin(std::vector<absl::string_view>(depth, "  "), "");
  if (VLOG_IS_ON(2)) {
    VLOG(2) << indent << "Visiting:";
    for (const auto& l : absl::StrSplit(to_print.DebugString(), '\n')) {
      VLOG(2) << indent << l;
    }
  }
  switch (opcode) {
    default:
      for (const int64_t operand_id : instr.operand_ids()) {
        IsConstantVisitor(operand_id, depth + 1, visited, is_constant);
      }
      // TODO(b/32495713): We aren't checking the called computations.
      break;

    case HloOpcode::kGetDimensionSize:
      // GetDimensionSize is always considered constant in XLA -- If a dynamic
      // dimension is presented, -1 is returned.
      break;
    // Non functional ops.
    case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      // TODO(b/33009255): Implement constant folding for cross replica sum.
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
      // TODO(b/32495713): We aren't checking the to_apply computation itself,
      // so we conservatively say that computations containing the Call op
      // cannot be constant.  We cannot set is_functional=false in other similar
      // cases since we're already relying on IsConstant to return true.
    case HloOpcode::kCustomCall:
      if (instr.custom_call_target() == "SetBound") {
        // Set bound is considered constant -- the bound is used as the value.
        break;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case HloOpcode::kWhile:
      // TODO(b/32495713): We aren't checking the condition and body
      // computations themselves.
    case HloOpcode::kScatter:
      // TODO(b/32495713): We aren't checking the embedded computation in
      // Scatter.
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter:
      *is_constant = false;
      break;
    case HloOpcode::kGetTupleElement: {
      const HloInstructionProto& operand_instr =
          *(LookUpInstructionByHandle(instr.operand_ids(0)).ValueOrDie());
      if (HloOpcodeString(HloOpcode::kTuple) == operand_instr.opcode()) {
        IsConstantVisitor(operand_instr.operand_ids(instr.tuple_index()),
                          depth + 1, visited, is_constant);
      } else {
        for (const int64_t operand_id : instr.operand_ids()) {
          IsConstantVisitor(operand_id, depth + 1, visited, is_constant);
        }
      }
    }
  }
  if (VLOG_IS_ON(1) && !*is_constant) {
    VLOG(1) << indent << "Non-constant: ";
    for (const auto& l : absl::StrSplit(to_print.DebugString(), '\n')) {
      VLOG(1) << indent << l;
    }
  }
  visited->insert(op_handle);
}

Status XlaBuilder::SetDynamicBinding(int64_t dynamic_size_param_num,
                                     ShapeIndex dynamic_size_param_index,
                                     int64_t target_param_num,
                                     ShapeIndex target_param_index,
                                     int64_t target_dim_num) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_36(mht_36_v, 694, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SetDynamicBinding");

  bool param_exists = false;
  for (size_t index = 0; index < instructions_.size(); ++index) {
    HloInstructionProto& instr = instructions_[index];
    if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter) &&
        instr.parameter_number() == target_param_num) {
      param_exists = true;
      Shape param_shape(instr.shape());
      Shape* param_shape_ptr = &param_shape;
      for (int64_t index : target_param_index) {
        param_shape_ptr = param_shape_ptr->mutable_tuple_shapes(index);
      }
      param_shape_ptr->set_dynamic_dimension(target_dim_num,
                                             /*is_dynamic=*/true);
      *instr.mutable_shape() = param_shape.ToProto();
      instruction_shapes_[index] =
          absl::make_unique<Shape>(std::move(param_shape));
    }
  }
  if (!param_exists) {
    return InvalidArgument(
        "Asked to mark parameter %lld as dynamic sized parameter, but the "
        "doesn't exists",
        target_param_num);
  }

  TF_RETURN_IF_ERROR(dynamic_parameter_binding_.Bind(
      DynamicParameterBinding::DynamicParameter{dynamic_size_param_num,
                                                dynamic_size_param_index},
      DynamicParameterBinding::DynamicDimension{
          target_param_num, target_param_index, target_dim_num}));
  return Status::OK();
}

Status XlaBuilder::SetInstructionFrontendAttribute(const XlaOp op,
                                                   std::string attribute,
                                                   std::string value) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("attribute: \"" + attribute + "\"");
   mht_37_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_37(mht_37_v, 735, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SetInstructionFrontendAttribute");

  TF_ASSIGN_OR_RETURN(auto instr_proto, LookUpMutableInstruction(op));
  auto* frontend_attributes = instr_proto->mutable_frontend_attributes();
  (*frontend_attributes->mutable_map())[attribute] = std::move(value);
  return Status::OK();
}

XlaComputation XlaBuilder::BuildAndNoteError() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_38(mht_38_v, 745, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BuildAndNoteError");

  DCHECK(parent_builder_ != nullptr);
  auto build_status = Build();
  if (!build_status.ok()) {
    parent_builder_->ReportError(
        AddStatus(build_status.status(), absl::StrCat("error from: ", name_)));
    return {};
  }
  return build_status.ConsumeValueOrDie();
}

Status XlaBuilder::GetCurrentStatus() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_39(mht_39_v, 759, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetCurrentStatus");

  if (!first_error_.ok()) {
    std::string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }
  return Status::OK();
}

StatusOr<XlaComputation> XlaBuilder::Build(bool remove_dynamic_dimensions) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_40(mht_40_v, 771, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Build");

  TF_RETURN_IF_ERROR(GetCurrentStatus());
  return Build(instructions_.back().id(), remove_dynamic_dimensions);
}

StatusOr<XlaComputation> XlaBuilder::Build(XlaOp root,
                                           bool remove_dynamic_dimensions) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_41(mht_41_v, 780, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Build");

  if (root.builder_ != this) {
    return InvalidArgument("Given root operation is not in this computation.");
  }
  return Build(root.handle(), remove_dynamic_dimensions);
}

StatusOr<XlaComputation> XlaBuilder::Build(int64_t root_id,
                                           bool remove_dynamic_dimensions) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_42(mht_42_v, 791, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Build");

  TF_RETURN_IF_ERROR(GetCurrentStatus());

  // TODO(b/121223198): XLA backend cannot handle dynamic dimensions yet, remove
  // all dynamic dimensions before building xla program until we have support in
  // the backend.
  if (remove_dynamic_dimensions) {
    std::function<void(Shape*)> remove_dynamic_dimension = [&](Shape* shape) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_43(mht_43_v, 801, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "lambda");

      if (shape->tuple_shapes_size() != 0) {
        for (int i = 0; i < shape->tuple_shapes_size(); ++i) {
          remove_dynamic_dimension(shape->mutable_tuple_shapes(i));
        }
      }
      for (int64_t i = 0; i < shape->dimensions_size(); ++i) {
        shape->set_dynamic_dimension(i, false);
      }
    };
    for (size_t index = 0; index < instructions_.size(); ++index) {
      remove_dynamic_dimension(instruction_shapes_[index].get());
      *instructions_[index].mutable_shape() =
          instruction_shapes_[index]->ToProto();
    }
  }

  HloComputationProto entry;
  SetProtoIdAndName(&entry, name_, kNameSeparator, GetNextId());
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, GetProgramShape(root_id));
  *entry.mutable_program_shape() = program_shape.ToProto();
  entry.set_root_id(root_id);

  for (auto& instruction : instructions_) {
    // Ensures that the instruction names are unique among the whole graph.
    instruction.set_name(
        GetFullName(instruction.name(), kNameSeparator, instruction.id()));
    entry.add_instructions()->Swap(&instruction);
  }

  XlaComputation computation(entry.id());
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
  *module->mutable_host_program_shape() = entry.program_shape();
  for (auto& e : embedded_) {
    module->add_computations()->Swap(&e.second);
  }
  module->add_computations()->Swap(&entry);
  if (!input_output_aliases_.empty()) {
    TF_RETURN_IF_ERROR(
        PopulateInputOutputAlias(module, program_shape, input_output_aliases_));
  }
  *(module->mutable_dynamic_parameter_binding()) =
      dynamic_parameter_binding_.ToProto();

  // Clear data held by this builder.
  this->instructions_.clear();
  this->instruction_shapes_.clear();
  this->handle_to_index_.clear();
  this->embedded_.clear();
  this->parameter_numbers_.clear();

  return std::move(computation);
}

/* static */ Status XlaBuilder::PopulateInputOutputAlias(
    HloModuleProto* module, const ProgramShape& program_shape,
    const std::vector<InputOutputAlias>& input_output_aliases) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_44(mht_44_v, 864, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::PopulateInputOutputAlias");

  HloInputOutputAliasConfig config(program_shape.result());
  for (auto& alias : input_output_aliases) {
    // The HloInputOutputAliasConfig does not do parameter validation as it only
    // carries the result shape. Maybe it should be constructed with a
    // ProgramShape to allow full validation. We will still get an error when
    // trying to compile the HLO module, but would be better to have validation
    // at this stage.
    if (alias.param_number >= program_shape.parameters_size()) {
      return InvalidArgument("Invalid parameter number %ld (total %ld)",
                             alias.param_number,
                             program_shape.parameters_size());
    }
    const Shape& parameter_shape = program_shape.parameters(alias.param_number);
    if (!ShapeUtil::IndexIsValid(parameter_shape, alias.param_index)) {
      return InvalidArgument("Invalid parameter %ld index: %s",
                             alias.param_number,
                             alias.param_index.ToString().c_str());
    }
    TF_RETURN_IF_ERROR(config.SetUpAlias(alias.output_index, alias.param_number,
                                         alias.param_index, alias.kind));
  }
  *module->mutable_input_output_alias() = config.ToProto();
  return Status::OK();
}

StatusOr<XlaOp> XlaBuilder::InDimBroadcast(
    const Shape& shape, XlaOp operand,
    absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_45(mht_45_v, 895, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::InDimBroadcast");

  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : broadcast_dimensions) {
    instr.add_dimensions(dim);
  }

  return AddInstruction(std::move(instr), HloOpcode::kBroadcast, {operand});
}

StatusOr<XlaOp> XlaBuilder::AddBroadcastSequence(const Shape& output_shape,
                                                 XlaOp operand) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_46(mht_46_v, 911, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AddBroadcastSequence");

  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

  CHECK(ShapeUtil::IsScalar(*operand_shape) ||
        operand_shape->rank() == output_shape.rank());
  Shape broadcast_shape =
      ShapeUtil::ChangeElementType(output_shape, operand_shape->element_type());

  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(*operand_shape)) {
    return InDimBroadcast(broadcast_shape, operand, {});
  }

  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64_t> broadcast_dimensions;
  std::vector<int64_t> reshaped_dimensions;
  for (int i = 0; i < operand_shape->rank(); i++) {
    if (operand_shape->dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand_shape->dimensions(i));
    } else {
      TF_RET_CHECK(operand_shape->dimensions(i) == 1)
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand shape: "
          << *operand_shape << "; output_shape: " << output_shape;
    }
  }

  Shape reshaped_shape =
      ShapeUtil::MakeShape(operand_shape->element_type(), reshaped_dimensions);

  std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
      ShapeUtil::DimensionsUnmodifiedByReshape(*operand_shape, reshaped_shape);

  for (auto& unmodified : unmodified_dims) {
    if (operand_shape->is_dynamic_dimension(unmodified.first)) {
      reshaped_shape.set_dynamic_dimension(unmodified.second, true);
    }
  }

  // Eliminate the size one dimensions.
  TF_ASSIGN_OR_RETURN(
      XlaOp reshaped_operand,
      ReshapeInternal(reshaped_shape, operand, /*inferred_dimension=*/-1));
  // Broadcast 'reshape' up to the larger size.
  return InDimBroadcast(broadcast_shape, reshaped_operand,
                        broadcast_dimensions);
}

XlaOp XlaBuilder::UnaryOp(HloOpcode unop, XlaOp operand) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_47(mht_47_v, 965, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::UnaryOp");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferUnaryOpShape(unop, *operand_shape));
    return AddOpWithShape(unop, shape, {operand});
  });
}

XlaOp XlaBuilder::BinaryOp(HloOpcode binop, XlaOp lhs, XlaOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions,
                           absl::optional<ComparisonDirection> direction,
                           absl::optional<Comparison::Type> type) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_48(mht_48_v, 980, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BinaryOp");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferBinaryOpShape(
                         binop, *lhs_shape, *rhs_shape, broadcast_dimensions));

    const int64_t lhs_rank = lhs_shape->rank();
    const int64_t rhs_rank = rhs_shape->rank();

    XlaOp updated_lhs = lhs;
    XlaOp updated_rhs = rhs;
    if (!broadcast_dimensions.empty() && lhs_rank != rhs_rank) {
      const bool should_broadcast_lhs = lhs_rank < rhs_rank;
      XlaOp from = should_broadcast_lhs ? lhs : rhs;
      const Shape& from_shape = should_broadcast_lhs ? *lhs_shape : *rhs_shape;

      std::vector<int64_t> to_size;
      std::vector<bool> to_size_is_dynamic;
      const auto rank = shape.rank();
      to_size.reserve(rank);
      to_size_is_dynamic.reserve(rank);
      for (int i = 0; i < rank; i++) {
        to_size.push_back(shape.dimensions(i));
        to_size_is_dynamic.push_back(shape.is_dynamic_dimension(i));
      }
      for (int64_t from_dim = 0; from_dim < from_shape.rank(); from_dim++) {
        int64_t to_dim = broadcast_dimensions[from_dim];
        to_size[to_dim] = from_shape.dimensions(from_dim);
        to_size_is_dynamic[to_dim] = from_shape.is_dynamic_dimension(from_dim);
      }

      const Shape& broadcasted_shape = ShapeUtil::MakeShape(
          from_shape.element_type(), to_size, to_size_is_dynamic);
      TF_ASSIGN_OR_RETURN(
          XlaOp broadcasted_operand,
          InDimBroadcast(broadcasted_shape, from, broadcast_dimensions));

      updated_lhs = should_broadcast_lhs ? broadcasted_operand : lhs;
      updated_rhs = !should_broadcast_lhs ? broadcasted_operand : rhs;
    }

    TF_ASSIGN_OR_RETURN(const Shape* updated_lhs_shape,
                        GetShapePtr(updated_lhs));
    if (!ShapeUtil::SameDimensions(shape, *updated_lhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_lhs,
                          AddBroadcastSequence(shape, updated_lhs));
    }
    TF_ASSIGN_OR_RETURN(const Shape* updated_rhs_shape,
                        GetShapePtr(updated_rhs));
    if (!ShapeUtil::SameDimensions(shape, *updated_rhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_rhs,
                          AddBroadcastSequence(shape, updated_rhs));
    }

    if (binop == HloOpcode::kCompare) {
      if (!direction.has_value()) {
        return InvalidArgument(
            "kCompare expects a ComparisonDirection, but none provided.");
      }
      if (type == absl::nullopt) {
        return Compare(shape, updated_lhs, updated_rhs, *direction);
      } else {
        return Compare(shape, updated_lhs, updated_rhs, *direction, *type);
      }
    }

    if (direction.has_value()) {
      return InvalidArgument(
          "A comparison direction is provided for a non-compare opcode: %s.",
          HloOpcodeString(binop));
    }
    return BinaryOpNoBroadcast(binop, shape, updated_lhs, updated_rhs);
  });
}

XlaOp XlaBuilder::BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                      XlaOp lhs, XlaOp rhs) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_49(mht_49_v, 1061, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BinaryOpNoBroadcast");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return AddInstruction(std::move(instr), binop, {lhs, rhs});
  });
}

StatusOr<XlaOp> XlaBuilder::Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                                    ComparisonDirection direction) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_50(mht_50_v, 1073, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Compare");

  TF_ASSIGN_OR_RETURN(auto operand_shape, GetShape(lhs));
  return Compare(
      shape, lhs, rhs, direction,
      Comparison::DefaultComparisonType(operand_shape.element_type()));
}

StatusOr<XlaOp> XlaBuilder::Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                                    ComparisonDirection direction,
                                    Comparison::Type type) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_51(mht_51_v, 1085, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Compare");

  HloInstructionProto instr;
  instr.set_comparison_direction(ComparisonDirectionToString(direction));
  instr.set_comparison_type(ComparisonTypeToString(type));
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kCompare, {lhs, rhs});
}

XlaOp XlaBuilder::TernaryOp(HloOpcode triop, XlaOp lhs, XlaOp rhs, XlaOp ehs) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_52(mht_52_v, 1096, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::TernaryOp");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    XlaOp updated_lhs = lhs;
    XlaOp updated_rhs = rhs;
    XlaOp updated_ehs = ehs;
    // The client API supports implicit broadcast for kSelect and kClamp, but
    // XLA does not support implicit broadcast. Make implicit broadcast explicit
    // and update the operands.
    if (triop == HloOpcode::kSelect || triop == HloOpcode::kClamp) {
      TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
      TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
      TF_ASSIGN_OR_RETURN(const Shape* ehs_shape, GetShapePtr(ehs));

      absl::optional<Shape> non_scalar_shape;
      for (const Shape* shape : {lhs_shape, rhs_shape, ehs_shape}) {
        if (shape->IsArray() && shape->rank() != 0) {
          if (non_scalar_shape.has_value()) {
            // TODO(jpienaar): The case where we need to compute the broadcasted
            // shape by considering multiple of the shapes is not implemented.
            // Consider reusing getBroadcastedType from mlir/Dialect/Traits.h.
            TF_RET_CHECK(non_scalar_shape.value().dimensions() ==
                         shape->dimensions())
                << "Unimplemented implicit broadcast.";
          } else {
            non_scalar_shape = *shape;
          }
        }
      }
      if (non_scalar_shape.has_value()) {
        if (ShapeUtil::IsScalar(*lhs_shape)) {
          TF_ASSIGN_OR_RETURN(updated_lhs,
                              AddBroadcastSequence(*non_scalar_shape, lhs));
        }
        if (ShapeUtil::IsScalar(*rhs_shape)) {
          TF_ASSIGN_OR_RETURN(updated_rhs,
                              AddBroadcastSequence(*non_scalar_shape, rhs));
        }
        if (ShapeUtil::IsScalar(*ehs_shape)) {
          TF_ASSIGN_OR_RETURN(updated_ehs,
                              AddBroadcastSequence(*non_scalar_shape, ehs));
        }
      }
    }

    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(updated_lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(updated_rhs));
    TF_ASSIGN_OR_RETURN(const Shape* ehs_shape, GetShapePtr(updated_ehs));
    StatusOr<const Shape> status_or_shape = ShapeInference::InferTernaryOpShape(
        triop, *lhs_shape, *rhs_shape, *ehs_shape);
    if (!status_or_shape.status().ok()) {
      return InvalidArgument(
          "%s Input scalar shapes may have been changed to non-scalar shapes.",
          status_or_shape.status().error_message());
    }

    return AddOpWithShape(triop, status_or_shape.ValueOrDie(),
                          {updated_lhs, updated_rhs, updated_ehs});
  });
}

XlaOp XlaBuilder::ConstantLiteral(const LiteralSlice& literal) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_53(mht_53_v, 1159, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConstantLiteral");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (literal.shape().IsArray() && literal.element_count() > 1 &&
        literal.IsAllFirst()) {
      Literal scalar = LiteralUtil::GetFirstScalarLiteral(literal);
      HloInstructionProto instr;
      *instr.mutable_shape() = scalar.shape().ToProto();
      *instr.mutable_literal() = scalar.ToProto();
      TF_ASSIGN_OR_RETURN(
          XlaOp scalar_op,
          AddInstruction(std::move(instr), HloOpcode::kConstant));
      return Broadcast(scalar_op, literal.shape().dimensions());
    } else {
      HloInstructionProto instr;
      *instr.mutable_shape() = literal.shape().ToProto();
      *instr.mutable_literal() = literal.ToProto();
      return AddInstruction(std::move(instr), HloOpcode::kConstant);
    }
  });
}

XlaOp XlaBuilder::Iota(const Shape& shape, int64_t iota_dimension) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_54(mht_54_v, 1183, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Iota");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    instr.add_dimensions(iota_dimension);
    return AddInstruction(std::move(instr), HloOpcode::kIota);
  });
}

XlaOp XlaBuilder::Iota(PrimitiveType type, int64_t size) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_55(mht_55_v, 1195, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Iota");

  return Iota(ShapeUtil::MakeShape(type, {size}), /*iota_dimension=*/0);
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_56(mht_56_v, 1203, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Call");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferCallShape(
                                         operand_shape_ptrs,
                                         /*to_apply=*/called_program_shape));
    *instr.mutable_shape() = shape.ToProto();

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kCall, operands);
  });
}

XlaOp XlaBuilder::Parameter(
    int64_t parameter_number, const Shape& shape, const std::string& name,
    const std::vector<bool>& replicated_at_leaf_buffers) {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_57(mht_57_v, 1229, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Parameter");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (!parameter_numbers_.insert(parameter_number).second) {
      return InvalidArgument("parameter %d already registered",
                             parameter_number);
    }
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape.ToProto();
    if (!replicated_at_leaf_buffers.empty()) {
      auto replication = instr.mutable_parameter_replication();
      for (bool replicated : replicated_at_leaf_buffers) {
        replication->add_replicated_at_leaf_buffers(replicated);
      }
    }
    return AddInstruction(std::move(instr), HloOpcode::kParameter);
  });
}

XlaOp XlaBuilder::Broadcast(XlaOp operand,
                            absl::Span<const int64_t> broadcast_sizes) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_58(mht_58_v, 1253, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Broadcast");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        const Shape& shape,
        ShapeInference::InferBroadcastShape(*operand_shape, broadcast_sizes));

    // The client-level broadcast op just appends dimensions on the left (adds
    // lowest numbered dimensions). The HLO broadcast instruction is more
    // flexible and can add new dimensions anywhere. The instruction's
    // dimensions field maps operand dimensions to dimensions in the broadcast
    // output, so to append dimensions on the left the instruction's dimensions
    // should just be the n highest dimension numbers of the output shape where
    // n is the number of input dimensions.
    const int64_t operand_rank = operand_shape->rank();
    std::vector<int64_t> dimensions(operand_rank);
    for (int i = 0; i < operand_rank; ++i) {
      dimensions[i] = i + shape.rank() - operand_rank;
    }
    return InDimBroadcast(shape, operand, dimensions);
  });
}

XlaOp XlaBuilder::BroadcastInDim(
    XlaOp operand, const absl::Span<const int64_t> out_dim_size,
    const absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_59(mht_59_v, 1281, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BroadcastInDim");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    // Output shape, in the case of degenerate broadcast, the out_dim_size is
    // not necessarily the same as the dimension sizes of the output shape.
    TF_ASSIGN_OR_RETURN(auto output_shape,
                        ShapeUtil::MakeValidatedShape(
                            operand_shape->element_type(), out_dim_size));
    int64_t broadcast_rank = broadcast_dimensions.size();
    if (operand_shape->rank() != broadcast_rank) {
      return InvalidArgument(
          "Size of broadcast_dimensions has to match operand's rank; operand "
          "rank: %lld, size of broadcast_dimensions %u.",
          operand_shape->rank(), broadcast_dimensions.size());
    }
    for (int i = 0; i < broadcast_rank; i++) {
      const int64_t num_dims = out_dim_size.size();
      if (broadcast_dimensions[i] < 0 || broadcast_dimensions[i] > num_dims) {
        return InvalidArgument("Broadcast dimension %lld is out of bound",
                               broadcast_dimensions[i]);
      }
      output_shape.set_dynamic_dimension(
          broadcast_dimensions[i], operand_shape->is_dynamic_dimension(i));
    }

    TF_RETURN_IF_ERROR(ShapeInference::InferBroadcastShape(
                           *operand_shape, output_shape, broadcast_dimensions)
                           .status());
    std::vector<int64_t> in_dim_size(out_dim_size.begin(), out_dim_size.end());
    for (int i = 0; i < broadcast_rank; i++) {
      in_dim_size[broadcast_dimensions[i]] = operand_shape->dimensions(i);
    }
    const auto& in_dim_shape =
        ShapeUtil::MakeShape(operand_shape->element_type(), in_dim_size);
    TF_ASSIGN_OR_RETURN(
        XlaOp in_dim_broadcast,
        InDimBroadcast(in_dim_shape, operand, broadcast_dimensions));

    // If broadcast is not degenerate, return broadcasted result.
    if (ShapeUtil::Equal(in_dim_shape, output_shape)) {
      return in_dim_broadcast;
    }

    // Otherwise handle degenerate broadcast case.
    return AddBroadcastSequence(output_shape, in_dim_broadcast);
  });
}

StatusOr<XlaOp> XlaBuilder::ReshapeInternal(const Shape& shape, XlaOp operand,
                                            int64_t inferred_dimension) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_60(mht_60_v, 1333, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReshapeInternal");

  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  if (inferred_dimension != -1) {
    instr.add_dimensions(inferred_dimension);
  }
  return AddInstruction(std::move(instr), HloOpcode::kReshape, {operand});
}

XlaOp XlaBuilder::Slice(XlaOp operand, absl::Span<const int64_t> start_indices,
                        absl::Span<const int64_t> limit_indices,
                        absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_61(mht_61_v, 1349, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Slice");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferSliceShape(
                                         *operand_shape, start_indices,
                                         limit_indices, strides));
    return SliceInternal(shape, operand, start_indices, limit_indices, strides);
  });
}

StatusOr<XlaOp> XlaBuilder::SliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_62(mht_62_v, 1365, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SliceInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int i = 0, end = start_indices.size(); i < end; i++) {
    auto* slice_config = instr.add_slice_dimensions();
    slice_config->set_start(start_indices[i]);
    slice_config->set_limit(limit_indices[i]);
    slice_config->set_stride(strides[i]);
  }
  return AddInstruction(std::move(instr), HloOpcode::kSlice, {operand});
}

XlaOp XlaBuilder::SliceInDim(XlaOp operand, int64_t start_index,
                             int64_t limit_index, int64_t stride,
                             int64_t dimno) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_63(mht_63_v, 1382, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SliceInDim");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    std::vector<int64_t> starts(shape->rank(), 0);
    std::vector<int64_t> limits(shape->dimensions().begin(),
                                shape->dimensions().end());
    std::vector<int64_t> strides(shape->rank(), 1);
    starts[dimno] = start_index;
    limits[dimno] = limit_index;
    strides[dimno] = stride;
    return Slice(operand, starts, limits, strides);
  });
}

XlaOp XlaBuilder::DynamicSlice(XlaOp operand,
                               absl::Span<const XlaOp> start_indices,
                               absl::Span<const int64_t> slice_sizes) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_64(mht_64_v, 1401, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicSlice");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> start_indices_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& start_indices_shapes,
                        GetOperandShapes(start_indices));
    absl::c_transform(start_indices_shapes,
                      std::back_inserter(start_indices_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferDynamicSliceShape(
                            *operand_shape, start_indices_shapes, slice_sizes));
    return DynamicSliceInternal(shape, operand, start_indices, slice_sizes);
  });
}

StatusOr<XlaOp> XlaBuilder::DynamicSliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
    absl::Span<const int64_t> slice_sizes) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_65(mht_65_v, 1422, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicSliceInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  for (int64_t size : slice_sizes) {
    instr.add_dynamic_slice_sizes(size);
  }

  std::vector<XlaOp> operands = {operand};
  operands.insert(operands.end(), start_indices.begin(), start_indices.end());
  return AddInstruction(std::move(instr), HloOpcode::kDynamicSlice, operands);
}

XlaOp XlaBuilder::DynamicUpdateSlice(XlaOp operand, XlaOp update,
                                     absl::Span<const XlaOp> start_indices) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_66(mht_66_v, 1439, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicUpdateSlice");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* update_shape, GetShapePtr(update));
    std::vector<const Shape*> start_indices_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& start_indices_shapes,
                        GetOperandShapes(start_indices));
    absl::c_transform(start_indices_shapes,
                      std::back_inserter(start_indices_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferDynamicUpdateSliceShape(
                         *operand_shape, *update_shape, start_indices_shapes));
    return DynamicUpdateSliceInternal(shape, operand, update, start_indices);
  });
}

StatusOr<XlaOp> XlaBuilder::DynamicUpdateSliceInternal(
    const Shape& shape, XlaOp operand, XlaOp update,
    absl::Span<const XlaOp> start_indices) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_67(mht_67_v, 1461, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicUpdateSliceInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  std::vector<XlaOp> operands = {operand, update};
  operands.insert(operands.end(), start_indices.begin(), start_indices.end());
  return AddInstruction(std::move(instr), HloOpcode::kDynamicUpdateSlice,
                        operands);
}

XlaOp XlaBuilder::ConcatInDim(absl::Span<const XlaOp> operands,
                              int64_t dimension) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_68(mht_68_v, 1475, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConcatInDim");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferConcatOpShape(
                                         operand_shape_ptrs, dimension));
    return ConcatInDimInternal(shape, operands, dimension);
  });
}

StatusOr<XlaOp> XlaBuilder::ConcatInDimInternal(
    const Shape& shape, absl::Span<const XlaOp> operands, int64_t dimension) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_69(mht_69_v, 1491, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConcatInDimInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  instr.add_dimensions(dimension);

  return AddInstruction(std::move(instr), HloOpcode::kConcatenate, operands);
}

XlaOp XlaBuilder::Pad(XlaOp operand, XlaOp padding_value,
                      const PaddingConfig& padding_config) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_70(mht_70_v, 1504, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Pad");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* padding_value_shape,
                        GetShapePtr(padding_value));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferPadShape(
                         *operand_shape, *padding_value_shape, padding_config));
    return PadInternal(shape, operand, padding_value, padding_config);
  });
}

XlaOp XlaBuilder::PadInDim(XlaOp operand, XlaOp padding_value, int64_t dimno,
                           int64_t pad_lo, int64_t pad_hi) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_71(mht_71_v, 1520, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::PadInDim");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    PaddingConfig padding_config = MakeNoPaddingConfig(shape->rank());
    auto* dims = padding_config.mutable_dimensions(dimno);
    dims->set_edge_padding_low(pad_lo);
    dims->set_edge_padding_high(pad_hi);
    return Pad(operand, padding_value, padding_config);
  });
}

StatusOr<XlaOp> XlaBuilder::PadInternal(const Shape& shape, XlaOp operand,
                                        XlaOp padding_value,
                                        const PaddingConfig& padding_config) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_72(mht_72_v, 1536, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::PadInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_padding_config() = padding_config;
  return AddInstruction(std::move(instr), HloOpcode::kPad,
                        {operand, padding_value});
}

XlaOp XlaBuilder::Reshape(XlaOp operand, absl::Span<const int64_t> dimensions,
                          absl::Span<const int64_t> new_sizes,
                          int64_t inferred_dimension) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_73(mht_73_v, 1549, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Reshape");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape shape, ShapeInference::InferReshapeShape(
                                               *operand_shape, dimensions,
                                               new_sizes, inferred_dimension));
    XlaOp transposed = IsIdentityPermutation(dimensions)
                           ? operand
                           : Transpose(operand, dimensions);
    return ReshapeInternal(shape, transposed, inferred_dimension);
  });
}

XlaOp XlaBuilder::Reshape(XlaOp operand, absl::Span<const int64_t> new_sizes,
                          int64_t inferred_dimension) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_74(mht_74_v, 1566, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Reshape");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    std::vector<int64_t> dimensions(shape->dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes, inferred_dimension);
  });
}

XlaOp XlaBuilder::Reshape(const Shape& shape, XlaOp operand,
                          int64_t inferred_dimension) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_75(mht_75_v, 1579, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Reshape");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return ReshapeInternal(shape, operand, inferred_dimension);
  });
}

XlaOp XlaBuilder::DynamicReshape(XlaOp operand,
                                 absl::Span<const XlaOp> dim_sizes,
                                 absl::Span<const int64_t> new_size_bounds,
                                 const std::vector<bool>& dims_are_dynamic) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_76(mht_76_v, 1591, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicReshape");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> dim_size_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& dim_size_shapes,
                        GetOperandShapes(dim_sizes));

    absl::c_transform(dim_size_shapes, std::back_inserter(dim_size_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferDynamicReshapeShape(
                            *operand_shape, dim_size_shape_ptrs,
                            new_size_bounds, dims_are_dynamic));
    TF_RETURN_IF_ERROR(first_error_);
    std::vector<XlaOp> operands;
    operands.reserve(1 + dim_sizes.size());
    operands.push_back(operand);
    for (const XlaOp& dim_size : dim_sizes) {
      operands.push_back(dim_size);
    }
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kDynamicReshape,
                          operands);
  });
}

XlaOp XlaBuilder::Collapse(XlaOp operand,
                           absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_77(mht_77_v, 1622, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Collapse");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (dimensions.size() <= 1) {
      // Not collapsing anything, trivially we can return the operand versus
      // enqueueing a trivial reshape.
      return operand;
    }

    // Out-of-order collapse is not supported.
    // Checks that the collapsed dimensions are in order and consecutive.
    for (absl::Span<const int64_t>::size_type i = 1; i < dimensions.size();
         ++i) {
      if (dimensions[i] - 1 != dimensions[i - 1]) {
        return InvalidArgument(
            "Collapsed dimensions are not in consecutive order.");
      }
    }

    // Create a new sizes vector from the old shape, replacing the collapsed
    // dimensions by the product of their sizes.
    TF_ASSIGN_OR_RETURN(const Shape* original_shape, GetShapePtr(operand));

    VLOG(3) << "original shape: " << ShapeUtil::HumanString(*original_shape);
    VLOG(3) << "dims to collapse: " << absl::StrJoin(dimensions, ",");

    std::vector<int64_t> new_sizes;
    for (int i = 0; i < original_shape->rank(); ++i) {
      if (i <= dimensions.front() || i > dimensions.back()) {
        new_sizes.push_back(original_shape->dimensions(i));
      } else {
        new_sizes.back() *= original_shape->dimensions(i);
      }
    }

    VLOG(3) << "new sizes: [" << absl::StrJoin(new_sizes, ",") << "]";

    return Reshape(operand, new_sizes);
  });
}

void XlaBuilder::Trace(const std::string& tag, XlaOp operand) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_78(mht_78_v, 1666, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Trace");

  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeNil().ToProto();
    *instr.mutable_literal() = LiteralUtil::CreateR1U8(tag).ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kTrace, {operand});
  });
}

XlaOp XlaBuilder::Select(XlaOp pred, XlaOp on_true, XlaOp on_false) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_79(mht_79_v, 1678, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Select");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* true_shape, GetShapePtr(on_true));
    TF_ASSIGN_OR_RETURN(const Shape* false_shape, GetShapePtr(on_false));
    TF_RET_CHECK(true_shape->IsTuple() == false_shape->IsTuple());
    HloOpcode opcode =
        true_shape->IsTuple() ? HloOpcode::kTupleSelect : HloOpcode::kSelect;
    return TernaryOp(opcode, pred, on_true, on_false);
  });
}

XlaOp XlaBuilder::Tuple(absl::Span<const XlaOp> elements) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_80(mht_80_v, 1692, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Tuple");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(elements));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferVariadicOpShape(
                            HloOpcode::kTuple, operand_shape_ptrs));
    return TupleInternal(shape, elements);
  });
}

StatusOr<XlaOp> XlaBuilder::TupleInternal(const Shape& shape,
                                          absl::Span<const XlaOp> elements) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_81(mht_81_v, 1709, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::TupleInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kTuple, elements);
}

XlaOp XlaBuilder::GetTupleElement(XlaOp tuple_data, int64_t index) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_82(mht_82_v, 1718, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetTupleElement");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* tuple_shape, GetShapePtr(tuple_data));
    if (!tuple_shape->IsTuple()) {
      return InvalidArgument(
          "Operand to GetTupleElement() is not a tuple; got %s",
          ShapeUtil::HumanString(*tuple_shape));
    }
    if (index < 0 || index >= ShapeUtil::TupleElementCount(*tuple_shape)) {
      return InvalidArgument(
          "GetTupleElement() index (%d) out of range for tuple shape %s", index,
          ShapeUtil::HumanString(*tuple_shape));
    }
    return GetTupleElementInternal(
        ShapeUtil::GetTupleElementShape(*tuple_shape, index), tuple_data,
        index);
  });
}

StatusOr<XlaOp> XlaBuilder::GetTupleElementInternal(const Shape& shape,
                                                    XlaOp tuple_data,
                                                    int64_t index) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_83(mht_83_v, 1742, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetTupleElementInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_tuple_index(index);
  return AddInstruction(std::move(instr), HloOpcode::kGetTupleElement,
                        {tuple_data});
}

XlaOp XlaBuilder::Dot(XlaOp lhs, XlaOp rhs,
                      const PrecisionConfig* precision_config,
                      absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_84(mht_84_v, 1755, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Dot");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));

    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape->dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    return DotGeneral(lhs, rhs, dimension_numbers, precision_config,
                      preferred_element_type);
  });
}

XlaOp XlaBuilder::DotGeneral(
    XlaOp lhs, XlaOp rhs, const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_85(mht_85_v, 1774, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DotGeneral");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferDotOpShape(
            *lhs_shape, *rhs_shape, dimension_numbers, preferred_element_type));
    return DotGeneralInternal(shape, lhs, rhs, dimension_numbers,
                              precision_config);
  });
}

StatusOr<XlaOp> XlaBuilder::DotGeneralInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig* precision_config) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_86(mht_86_v, 1793, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DotGeneralInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_dot_dimension_numbers() = dimension_numbers;
  if (precision_config != nullptr) {
    *instr.mutable_precision_config() = *precision_config;
  }
  return AddInstruction(std::move(instr), HloOpcode::kDot, {lhs, rhs});
}

Status XlaBuilder::VerifyConvolution(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const ConvolutionDimensionNumbers& dimension_numbers) const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_87(mht_87_v, 1808, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::VerifyConvolution");

  if (lhs_shape.rank() != rhs_shape.rank()) {
    return InvalidArgument(
        "Convolution arguments must have same number of "
        "dimensions. Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape), ShapeUtil::HumanString(rhs_shape));
  }
  int num_dims = lhs_shape.rank();
  if (num_dims < 2) {
    return InvalidArgument(
        "Convolution expects argument arrays with >= 3 dimensions. "
        "Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape), ShapeUtil::HumanString(rhs_shape));
  }
  int num_spatial_dims = num_dims - 2;

  const auto check_spatial_dimensions = [&](absl::string_view field_name,
                                            absl::Span<const int64_t> numbers) {
   std::vector<std::string> mht_88_v;
   mht_88_v.push_back("field_name: \"" + std::string(field_name.data(), field_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_88(mht_88_v, 1829, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "lambda");

    if (numbers.size() != num_spatial_dims) {
      return InvalidArgument("Expected %d elements for %s, but got %d.",
                             num_spatial_dims, field_name, numbers.size());
    }
    for (int i = 0; i < numbers.size(); ++i) {
      if (numbers[i] < 0 || numbers[i] >= num_dims) {
        return InvalidArgument("Convolution %s[%d] is out of bounds: %d",
                               field_name, i, numbers[i]);
      }
    }
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(
      check_spatial_dimensions("input_spatial_dimensions",
                               dimension_numbers.input_spatial_dimensions()));
  TF_RETURN_IF_ERROR(
      check_spatial_dimensions("kernel_spatial_dimensions",
                               dimension_numbers.kernel_spatial_dimensions()));
  return check_spatial_dimensions(
      "output_spatial_dimensions",
      dimension_numbers.output_spatial_dimensions());
}

XlaOp XlaBuilder::Conv(XlaOp lhs, XlaOp rhs,
                       absl::Span<const int64_t> window_strides,
                       Padding padding, int64_t feature_group_count,
                       int64_t batch_group_count,
                       const PrecisionConfig* precision_config,
                       absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_89(mht_89_v, 1861, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Conv");

  return ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding,
      CreateDefaultConvDimensionNumbers(window_strides.size()),
      feature_group_count, batch_group_count, precision_config,
      preferred_element_type);
}

XlaOp XlaBuilder::ConvWithGeneralPadding(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_90(mht_90_v, 1877, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvWithGeneralPadding");

  return ConvGeneral(lhs, rhs, window_strides, padding,
                     CreateDefaultConvDimensionNumbers(window_strides.size()),
                     feature_group_count, batch_group_count, precision_config,
                     preferred_element_type);
}

XlaOp XlaBuilder::ConvWithGeneralDimensions(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_91(mht_91_v, 1892, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvWithGeneralDimensions");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));

    TF_RETURN_IF_ERROR(
        VerifyConvolution(*lhs_shape, *rhs_shape, dimension_numbers));

    std::vector<int64_t> base_area_dimensions(
        dimension_numbers.input_spatial_dimensions_size());
    for (std::vector<int64_t>::size_type i = 0; i < base_area_dimensions.size();
         ++i) {
      base_area_dimensions[i] =
          lhs_shape->dimensions(dimension_numbers.input_spatial_dimensions(i));
    }

    std::vector<int64_t> window_dimensions(
        dimension_numbers.kernel_spatial_dimensions_size());
    for (std::vector<int64_t>::size_type i = 0; i < window_dimensions.size();
         ++i) {
      window_dimensions[i] =
          rhs_shape->dimensions(dimension_numbers.kernel_spatial_dimensions(i));
    }

    return ConvGeneral(lhs, rhs, window_strides,
                       MakePadding(base_area_dimensions, window_dimensions,
                                   window_strides, padding),
                       dimension_numbers, feature_group_count,
                       batch_group_count, precision_config,
                       preferred_element_type);
  });
}

XlaOp XlaBuilder::ConvGeneral(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_92(mht_92_v, 1934, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvGeneral");

  return ConvGeneralDilated(lhs, rhs, window_strides, padding, {}, {},
                            dimension_numbers, feature_group_count,
                            batch_group_count, precision_config,
                            preferred_element_type);
}

XlaOp XlaBuilder::ConvGeneralDilated(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_93(mht_93_v, 1952, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvGeneralDilated");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
    TF_RETURN_IF_ERROR(
        VerifyConvolution(*lhs_shape, *rhs_shape, dimension_numbers));

    std::vector<int64_t> window_dimensions(
        dimension_numbers.kernel_spatial_dimensions_size());
    for (std::vector<int64_t>::size_type i = 0; i < window_dimensions.size();
         ++i) {
      window_dimensions[i] =
          rhs_shape->dimensions(dimension_numbers.kernel_spatial_dimensions(i));
    }

    TF_ASSIGN_OR_RETURN(Window window,
                        ShapeInference::InferWindowFromDimensions(
                            window_dimensions, window_strides, padding,
                            lhs_dilation, rhs_dilation));
    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferConvolveShape(
            *lhs_shape, *rhs_shape, feature_group_count, batch_group_count,
            window, dimension_numbers, preferred_element_type));
    return ConvGeneralDilatedInternal(shape, lhs, rhs, window, window_strides,
                                      padding, lhs_dilation, rhs_dilation,
                                      dimension_numbers, feature_group_count,
                                      batch_group_count, precision_config);
  });
}

StatusOr<HloInstructionProto> XlaBuilder::DynamicConvInstruction(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_94(mht_94_v, 1994, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicConvInstruction");

  TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
  TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
  std::vector<int64_t> window_dimensions(
      dimension_numbers.kernel_spatial_dimensions_size());
  for (std::vector<int64_t>::size_type i = 0; i < window_dimensions.size();
       ++i) {
    window_dimensions[i] =
        rhs_shape->dimensions(dimension_numbers.kernel_spatial_dimensions(i));
  }

  TF_ASSIGN_OR_RETURN(Window window, ShapeInference::InferWindowFromDimensions(
                                         window_dimensions, window_strides,
                                         padding, lhs_dilation, rhs_dilation));
  TF_ASSIGN_OR_RETURN(
      Shape shape,
      ShapeInference::InferConvolveShape(
          *lhs_shape, *rhs_shape, feature_group_count, batch_group_count,
          window, dimension_numbers, preferred_element_type));

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  *instr.mutable_window() = window;
  *instr.mutable_convolution_dimension_numbers() = dimension_numbers;
  instr.set_feature_group_count(feature_group_count);
  instr.set_batch_group_count(batch_group_count);
  instr.set_padding_type(padding_type);

  if (precision_config != nullptr) {
    *instr.mutable_precision_config() = *precision_config;
  }
  return std::move(instr);
}

XlaOp XlaBuilder::DynamicConvInputGrad(
    XlaOp input_sizes, XlaOp lhs, XlaOp rhs,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_95(mht_95_v, 2041, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicConvInputGrad");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(
        HloInstructionProto instr,
        DynamicConvInstruction(
            lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
            dimension_numbers, feature_group_count, batch_group_count,
            precision_config, padding_type, preferred_element_type));

    instr.set_custom_call_target("DynamicConvolutionInputGrad");

    return AddInstruction(std::move(instr), HloOpcode::kCustomCall,
                          {input_sizes, lhs, rhs});
  });
}

XlaOp XlaBuilder::DynamicConvKernelGrad(
    XlaOp activations, XlaOp gradients,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_96(mht_96_v, 2069, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicConvKernelGrad");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(
        HloInstructionProto instr,
        DynamicConvInstruction(activations, gradients, window_strides, padding,
                               lhs_dilation, rhs_dilation, dimension_numbers,
                               feature_group_count, batch_group_count,
                               precision_config, padding_type,
                               preferred_element_type));

    instr.set_custom_call_target("DynamicConvolutionKernelGrad");
    // The gradient of kernel has kernel shape and shouldn't have any dynamic
    // sizes.
    instr.mutable_shape()->clear_is_dynamic_dimension();
    return AddInstruction(std::move(instr), HloOpcode::kCustomCall,
                          {activations, gradients});
  });
}

XlaOp XlaBuilder::DynamicConvForward(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_97(mht_97_v, 2099, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::DynamicConvForward");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(
        HloInstructionProto instr,
        DynamicConvInstruction(
            lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
            dimension_numbers, feature_group_count, batch_group_count,
            precision_config, padding_type, preferred_element_type));
    instr.set_custom_call_target("DynamicConvolutionForward");

    return AddInstruction(std::move(instr), HloOpcode::kCustomCall, {lhs, rhs});
  });
}

StatusOr<XlaOp> XlaBuilder::ConvGeneralDilatedInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_98(mht_98_v, 2124, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvGeneralDilatedInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();

  *instr.mutable_window() = window;
  *instr.mutable_convolution_dimension_numbers() = dimension_numbers;
  instr.set_feature_group_count(feature_group_count);
  instr.set_batch_group_count(batch_group_count);

  if (precision_config != nullptr) {
    *instr.mutable_precision_config() = *precision_config;
  }

  return AddInstruction(std::move(instr), HloOpcode::kConvolution, {lhs, rhs});
}

XlaOp XlaBuilder::Fft(XlaOp operand, const FftType fft_type,
                      const absl::Span<const int64_t> fft_length) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_99(mht_99_v, 2144, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Fft");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferFftShape(
                                         *operand_shape, fft_type, fft_length));
    return FftInternal(shape, operand, fft_type, fft_length);
  });
}

StatusOr<XlaOp> XlaBuilder::FftInternal(
    const Shape& shape, XlaOp operand, const FftType fft_type,
    const absl::Span<const int64_t> fft_length) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_100(mht_100_v, 2158, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::FftInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_fft_type(fft_type);
  for (int64_t i : fft_length) {
    instr.add_fft_length(i);
  }

  return AddInstruction(std::move(instr), HloOpcode::kFft, {operand});
}

StatusOr<XlaOp> XlaBuilder::TriangularSolveInternal(
    const Shape& shape, XlaOp a, XlaOp b, TriangularSolveOptions options) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_101(mht_101_v, 2173, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::TriangularSolveInternal");

  HloInstructionProto instr;
  *instr.mutable_triangular_solve_options() = std::move(options);
  *instr.mutable_shape() = shape.ToProto();

  return AddInstruction(std::move(instr), HloOpcode::kTriangularSolve, {a, b});
}

StatusOr<XlaOp> XlaBuilder::CholeskyInternal(const Shape& shape, XlaOp a,
                                             bool lower) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_102(mht_102_v, 2185, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CholeskyInternal");

  HloInstructionProto instr;
  xla::CholeskyOptions& options = *instr.mutable_cholesky_options();
  options.set_lower(lower);
  *instr.mutable_shape() = shape.ToProto();

  return AddInstruction(std::move(instr), HloOpcode::kCholesky, {a});
}

XlaOp XlaBuilder::Infeed(const Shape& shape, const std::string& config) {
   std::vector<std::string> mht_103_v;
   mht_103_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_103(mht_103_v, 2198, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Infeed");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Given shape to Infeed must have a layout");
    }
    const Shape infeed_instruction_shape =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});
    *instr.mutable_shape() = infeed_instruction_shape.ToProto();
    instr.set_infeed_config(config);

    if (shape.IsArray() && sharding() &&
        sharding()->type() == OpSharding::OTHER) {
      // TODO(b/110793772): Support tiled array-shaped infeeds.
      return InvalidArgument(
          "Tiled sharding is not yet supported for array-shaped infeeds");
    }

    if (sharding() && sharding()->type() == OpSharding::REPLICATED) {
      return InvalidArgument(
          "Replicated sharding is not yet supported for infeeds");
    }

    // Infeed takes a single token operand. Generate the token to pass to the
    // infeed.
    XlaOp token;
    auto make_token = [&]() {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_104(mht_104_v, 2227, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "lambda");

      HloInstructionProto token_instr;
      *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
      return AddInstruction(std::move(token_instr), HloOpcode::kAfterAll, {});
    };
    if (sharding()) {
      // Arbitrarily assign token to device 0.
      OpSharding sharding = sharding_builder::AssignDevice(0);
      XlaScopedShardingAssignment scoped_sharding(this, sharding);
      TF_ASSIGN_OR_RETURN(token, make_token());
    } else {
      TF_ASSIGN_OR_RETURN(token, make_token());
    }

    // The sharding is set by the client according to the data tuple shape.
    // However, the shape of the infeed instruction is a tuple containing the
    // data and a token. For tuple sharding type, the sharding must be changed
    // to accommodate the token.
    XlaOp infeed;
    if (sharding() && sharding()->type() == OpSharding::TUPLE) {
      // TODO(b/80000000): Remove this when clients have been updated to handle
      // tokens.
      OpSharding infeed_instruction_sharding = *sharding();
      // Arbitrarily assign the token to device 0.
      *infeed_instruction_sharding.add_tuple_shardings() =
          sharding_builder::AssignDevice(0);
      XlaScopedShardingAssignment scoped_sharding(this,
                                                  infeed_instruction_sharding);
      TF_ASSIGN_OR_RETURN(infeed, AddInstruction(std::move(instr),
                                                 HloOpcode::kInfeed, {token}));
    } else {
      TF_ASSIGN_OR_RETURN(infeed, AddInstruction(std::move(instr),
                                                 HloOpcode::kInfeed, {token}));
    }

    // The infeed instruction produces a tuple of the infed data and a token
    // type. Return XLA op containing the data.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto infeed_data;
    *infeed_data.mutable_shape() = shape.ToProto();
    infeed_data.set_tuple_index(0);
    return AddInstruction(std::move(infeed_data), HloOpcode::kGetTupleElement,
                          {infeed});
  });
}

XlaOp XlaBuilder::InfeedWithToken(XlaOp token, const Shape& shape,
                                  const std::string& config) {
   std::vector<std::string> mht_105_v;
   mht_105_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_105(mht_105_v, 2279, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::InfeedWithToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Given shape to Infeed must have a layout");
    }
    const Shape infeed_instruction_shape =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});

    if (shape.IsArray() && sharding() &&
        sharding()->type() == OpSharding::OTHER) {
      // TODO(b/110793772): Support tiled array-shaped infeeds.
      return InvalidArgument(
          "Tiled sharding is not yet supported for array-shaped infeeds");
    }

    if (sharding() && sharding()->type() == OpSharding::REPLICATED) {
      return InvalidArgument(
          "Replicated sharding is not yet supported for infeeds");
    }
    return InfeedWithTokenInternal(infeed_instruction_shape, token, config);
  });
}

StatusOr<XlaOp> XlaBuilder::InfeedWithTokenInternal(
    const Shape& infeed_instruction_shape, XlaOp token,
    const std::string& config) {
   std::vector<std::string> mht_106_v;
   mht_106_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_106(mht_106_v, 2308, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::InfeedWithTokenInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = infeed_instruction_shape.ToProto();
  instr.set_infeed_config(config);
  return AddInstruction(std::move(instr), HloOpcode::kInfeed, {token});
}

void XlaBuilder::Outfeed(XlaOp operand, const Shape& shape_with_layout,
                         const std::string& outfeed_config) {
   std::vector<std::string> mht_107_v;
   mht_107_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_107(mht_107_v, 2320, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Outfeed");

  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    *instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();

    // Check and set outfeed shape.
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Given shape to Outfeed must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    if (!ShapeUtil::Compatible(*operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "Outfeed shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(*operand_shape));
    }
    *instr.mutable_outfeed_shape() = shape_with_layout.ToProto();

    instr.set_outfeed_config(outfeed_config);

    // Outfeed takes a token as its second operand. Generate the token to pass
    // to the outfeed.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    TF_RETURN_IF_ERROR(
        AddInstruction(std::move(instr), HloOpcode::kOutfeed, {operand, token})
            .status());

    // The outfeed instruction produces a token. However, existing users expect
    // a nil shape (empty tuple). This should only be relevant if the outfeed is
    // the root of a computation.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto tuple_instr;
    *tuple_instr.mutable_shape() = ShapeUtil::MakeNil().ToProto();

    // The dummy tuple should have no sharding.
    {
      XlaScopedShardingAssignment scoped_sharding(this, OpSharding());
      TF_ASSIGN_OR_RETURN(
          XlaOp empty_tuple,
          AddInstruction(std::move(tuple_instr), HloOpcode::kTuple, {}));
      return empty_tuple;
    }
  });
}

XlaOp XlaBuilder::OutfeedWithToken(XlaOp operand, XlaOp token,
                                   const Shape& shape_with_layout,
                                   const std::string& outfeed_config) {
   std::vector<std::string> mht_108_v;
   mht_108_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_108(mht_108_v, 2377, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::OutfeedWithToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Check and set outfeed shape.
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Given shape to Outfeed must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    if (!ShapeUtil::Compatible(*operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "Outfeed shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(*operand_shape));
    }
    return OutfeedWithTokenInternal(operand, token, shape_with_layout,
                                    outfeed_config);
  });
}

StatusOr<XlaOp> XlaBuilder::OutfeedWithTokenInternal(
    XlaOp operand, XlaOp token, const Shape& shape_with_layout,
    const std::string& outfeed_config) {
   std::vector<std::string> mht_109_v;
   mht_109_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_109(mht_109_v, 2401, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::OutfeedWithTokenInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
  *instr.mutable_outfeed_shape() = shape_with_layout.ToProto();
  instr.set_outfeed_config(outfeed_config);
  return AddInstruction(std::move(instr), HloOpcode::kOutfeed,
                        {operand, token});
}

XlaOp XlaBuilder::CreateToken() {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_110(mht_110_v, 2413, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CreateToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kAfterAll);
  });
}

XlaOp XlaBuilder::AfterAll(absl::Span<const XlaOp> tokens) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_111(mht_111_v, 2424, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AfterAll");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (tokens.empty()) {
      return InvalidArgument("AfterAll requires at least one operand");
    }
    for (int i = 0, end = tokens.size(); i < end; ++i) {
      XlaOp operand = tokens[i];
      TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
      if (!operand_shape->IsToken()) {
        return InvalidArgument(
            "All operands to AfterAll must be tokens; operand %d has shape %s",
            i, ShapeUtil::HumanString(*operand_shape));
      }
    }
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kAfterAll, tokens);
  });
}

XlaOp XlaBuilder::CustomCall(
    const std::string& call_target_name, absl::Span<const XlaOp> operands,
    const Shape& shape, const std::string& opaque,
    absl::optional<absl::Span<const Shape>> operand_shapes_with_layout,
    bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, absl::optional<Window> window,
    absl::optional<ConvolutionDimensionNumbers> dnums,
    CustomCallSchedule schedule, CustomCallApiVersion api_version) {
   std::vector<std::string> mht_112_v;
   mht_112_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_112_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_112(mht_112_v, 2458, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CustomCall");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (absl::StartsWith(call_target_name, "$")) {
      return InvalidArgument(
          "Invalid custom_call_target \"%s\": Call targets that start with '$' "
          "are reserved for internal use.",
          call_target_name);
    }
    if (operand_shapes_with_layout.has_value()) {
      if (!LayoutUtil::HasLayout(shape)) {
        return InvalidArgument(
            "Result shape must have layout for custom call with constrained "
            "layout.");
      }
      if (operands.size() != operand_shapes_with_layout->size()) {
        return InvalidArgument(
            "Must specify a shape with layout for each operand for custom call "
            "with constrained layout; given %d shapes, expected %d",
            operand_shapes_with_layout->size(), operands.size());
      }
      int64_t operand_num = 0;
      for (const Shape& operand_shape : *operand_shapes_with_layout) {
        if (!LayoutUtil::HasLayout(operand_shape)) {
          return InvalidArgument(
              "No layout specified for operand %d for custom call with "
              "constrained layout.",
              operand_num);
        }
        ++operand_num;
      }
    }
    return CustomCallInternal(call_target_name, operands, shape, opaque,
                              operand_shapes_with_layout, has_side_effect,
                              output_operand_aliasing, literal, window, dnums,
                              schedule, api_version);
  });
}

StatusOr<XlaOp> XlaBuilder::CustomCallInternal(
    const std::string& call_target_name, absl::Span<const XlaOp> operands,
    const Shape& shape, const std::string& opaque,
    absl::optional<absl::Span<const Shape>> operand_shapes_with_layout,
    bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, absl::optional<Window> window,
    absl::optional<ConvolutionDimensionNumbers> dnums,
    CustomCallSchedule schedule, CustomCallApiVersion api_version) {
   std::vector<std::string> mht_113_v;
   mht_113_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_113_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_113(mht_113_v, 2510, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CustomCallInternal");

  HloInstructionProto instr;
  // Bit of a hack: cudnn conv custom-calls are created through this API. Give
  // them a user-friendly name. (This has no effect on correctness, it's just
  // cosmetic.)
  if (call_target_name == "__cudnn$convForward") {
    instr.set_name("cudnn-conv");
  } else if (call_target_name == "__cudnn$convBackwardInput") {
    instr.set_name("cudnn-conv-bw-input");
  } else if (call_target_name == "__cudnn$convBackwardFilter") {
    instr.set_name("cudnn-conv-bw-filter");
  } else if (call_target_name == "__cudnn$convBiasActivationForward") {
    instr.set_name("cudnn-conv-bias-activation");
  }
  *instr.mutable_shape() = shape.ToProto();
  instr.set_custom_call_target(call_target_name);
  instr.set_backend_config(opaque);
  if (operand_shapes_with_layout.has_value()) {
    instr.set_constrain_layout(true);
    for (const Shape& operand_shape : *operand_shapes_with_layout) {
      *instr.add_operand_shapes_with_layout() = operand_shape.ToProto();
    }
  }
  if (literal != nullptr) {
    *instr.mutable_literal() = literal->ToProto();
  }
  instr.set_custom_call_has_side_effect(has_side_effect);
  for (const auto& pair : output_operand_aliasing) {
    auto aliasing = instr.add_custom_call_output_operand_aliasing();
    aliasing->set_operand_index(pair.second.first);
    for (int64_t index : pair.second.second) {
      aliasing->add_operand_shape_index(index);
    }
    for (int64_t index : pair.first) {
      aliasing->add_output_shape_index(index);
    }
  }
  if (window.has_value()) {
    *instr.mutable_window() = *window;
  }
  if (dnums.has_value()) {
    *instr.mutable_convolution_dimension_numbers() = *dnums;
  }
  instr.set_custom_call_schedule(schedule);
  instr.set_custom_call_api_version(api_version);
  return AddInstruction(std::move(instr), HloOpcode::kCustomCall, operands);
}

XlaOp XlaBuilder::CustomCall(
    const std::string& call_target_name, absl::Span<const XlaOp> operands,
    const XlaComputation& computation, const Shape& shape,
    const std::string& opaque,
    absl::optional<absl::Span<const Shape>> operand_shapes_with_layout,
    bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, CustomCallSchedule schedule,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_114_v;
   mht_114_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_114_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_114(mht_114_v, 2572, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CustomCall");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (absl::StartsWith(call_target_name, "$")) {
      return InvalidArgument(
          "Invalid custom_call_target \"%s\": Call targets that start with '$' "
          "are reserved for internal use.",
          call_target_name);
    }
    *instr.mutable_shape() = shape.ToProto();
    instr.set_custom_call_target(call_target_name);
    instr.set_backend_config(opaque);
    if (literal != nullptr) {
      *instr.mutable_literal() = literal->ToProto();
    }
    if (operand_shapes_with_layout.has_value()) {
      if (!LayoutUtil::HasLayout(shape)) {
        return InvalidArgument(
            "Result shape must have layout for custom call with constrained "
            "layout.");
      }
      if (operands.size() != operand_shapes_with_layout->size()) {
        return InvalidArgument(
            "Must specify a shape with layout for each operand for custom call "
            "with constrained layout; given %d shapes, expected %d",
            operand_shapes_with_layout->size(), operands.size());
      }
      instr.set_constrain_layout(true);
      int64_t operand_num = 0;
      for (const Shape& operand_shape : *operand_shapes_with_layout) {
        if (!LayoutUtil::HasLayout(operand_shape)) {
          return InvalidArgument(
              "No layout specified for operand %d for custom call with "
              "constrained layout.",
              operand_num);
        }
        *instr.add_operand_shapes_with_layout() = operand_shape.ToProto();
        ++operand_num;
      }
    }
    AddCalledComputation(computation, &instr);
    for (const auto& pair : output_operand_aliasing) {
      auto aliasing = instr.add_custom_call_output_operand_aliasing();
      aliasing->set_operand_index(pair.second.first);
      for (int64_t index : pair.second.second) {
        aliasing->add_operand_shape_index(index);
      }
      for (int64_t index : pair.first) {
        aliasing->add_output_shape_index(index);
      }
    }
    instr.set_custom_call_schedule(schedule);
    instr.set_custom_call_api_version(api_version);
    return AddInstruction(std::move(instr), HloOpcode::kCustomCall, operands);
  });
}

XlaOp XlaBuilder::OptimizationBarrier(XlaOp operand) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_115(mht_115_v, 2632, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::OptimizationBarrier");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    Shape shape = *operand_shape;
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kOptimizationBarrier,
                          {operand});
  });
}

XlaOp XlaBuilder::Transpose(XlaOp operand,
                            absl::Span<const int64_t> permutation) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_116(mht_116_v, 2647, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Transpose");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferTransposeShape(
                                         *operand_shape, permutation));
    return TransposeInternal(shape, operand, permutation);
  });
}

StatusOr<XlaOp> XlaBuilder::TransposeInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> permutation) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_117(mht_117_v, 2660, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::TransposeInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : permutation) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kTranspose, {operand});
}

XlaOp XlaBuilder::Rev(XlaOp operand, absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_118(mht_118_v, 2672, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Rev");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferReverseShape(
                                         *operand_shape, dimensions));
    return RevInternal(shape, operand, dimensions);
  });
}

StatusOr<XlaOp> XlaBuilder::RevInternal(const Shape& shape, XlaOp operand,
                                        absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_119(mht_119_v, 2685, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RevInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  for (int64_t dim : dimensions) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kReverse, {operand});
}

XlaOp XlaBuilder::Sort(absl::Span<const XlaOp> operands,
                       const XlaComputation& comparator, int64_t dimension,
                       bool is_stable) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_120(mht_120_v, 2699, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Sort");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(std::vector<Shape> operand_shapes,
                        GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferVariadicOpShape(
                                         HloOpcode::kSort, operand_shape_ptrs));
    return SortInternal(shape, operands, comparator, dimension, is_stable);
  });
}

StatusOr<XlaOp> XlaBuilder::SortInternal(const Shape& shape,
                                         absl::Span<const XlaOp> operands,
                                         const XlaComputation& comparator,
                                         int64_t dimension, bool is_stable) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_121(mht_121_v, 2718, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SortInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_is_stable(is_stable);
  if (dimension == -1) {
    TF_ASSIGN_OR_RETURN(const Shape* keys_shape, GetShapePtr(operands[0]));
    dimension = keys_shape->rank() - 1;
  }
  instr.add_dimensions(dimension);
  AddCalledComputation(comparator, &instr);
  return AddInstruction(std::move(instr), HloOpcode::kSort, operands);
}

XlaOp XlaBuilder::ConvertElementType(XlaOp operand,
                                     PrimitiveType new_element_type) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_122(mht_122_v, 2735, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConvertElementType");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferConvertShape(
                                         *operand_shape, new_element_type));
    return AddOpWithShape(HloOpcode::kConvert, shape, {operand});
  });
}

XlaOp XlaBuilder::BitcastConvertType(XlaOp operand,
                                     PrimitiveType new_element_type) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_123(mht_123_v, 2748, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BitcastConvertType");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferBitcastConvertShape(
                                         *operand_shape, new_element_type));
    return BitcastConvertTypeInternal(shape, operand);
  });
}

StatusOr<XlaOp> XlaBuilder::BitcastConvertTypeInternal(const Shape& shape,
                                                       XlaOp operand) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_124(mht_124_v, 2761, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BitcastConvertTypeInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kBitcastConvert,
                        {operand});
}

XlaOp XlaBuilder::Clamp(XlaOp min, XlaOp operand, XlaOp max) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_125(mht_125_v, 2771, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Clamp");

  return TernaryOp(HloOpcode::kClamp, min, operand, max);
}

XlaOp XlaBuilder::Map(absl::Span<const XlaOp> operands,
                      const XlaComputation& computation,
                      absl::Span<const int64_t> dimensions,
                      absl::Span<const XlaOp> static_operands) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_126(mht_126_v, 2781, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Map");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!static_operands.empty()) {
      return Unimplemented("static_operands is not supported in Map");
    }

    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferMapShape(
                         operand_shape_ptrs, called_program_shape, dimensions));
    *instr.mutable_shape() = shape.ToProto();

    Shape output_shape(instr.shape());
    const int64_t output_rank = output_shape.rank();
    AddCalledComputation(computation, &instr);
    std::vector<XlaOp> new_operands(operands.begin(), operands.end());
    for (XlaOp& new_operand : new_operands) {
      TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(new_operand));
      const int64_t rank = shape->rank();
      if (rank != output_rank) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            InDimBroadcast(output_shape, new_operand, {}));
        TF_ASSIGN_OR_RETURN(shape, GetShapePtr(new_operand));
      }
      if (!ShapeUtil::SameDimensions(output_shape, *shape)) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            AddBroadcastSequence(output_shape, new_operand));
      }
    }

    return AddInstruction(std::move(instr), HloOpcode::kMap, new_operands);
  });
}

XlaOp XlaBuilder::RngOp(RandomDistribution distribution,
                        absl::Span<const XlaOp> parameters,
                        const Shape& shape) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_127(mht_127_v, 2826, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngOp");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Check the number of parameters per RNG distribution.
    switch (distribution) {
      case RandomDistribution::RNG_NORMAL:
      case RandomDistribution::RNG_UNIFORM:
        if (parameters.size() != 2) {
          return InvalidArgument(
              "RNG distribution (%s) expects 2 parameters, but got %ld",
              RandomDistribution_Name(distribution), parameters.size());
        }
        break;
      default:
        LOG(FATAL) << "unhandled distribution " << distribution;
    }

    TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
    return RngOpInternal(distribution, parameters, shape);
  });
}

StatusOr<XlaOp> XlaBuilder::RngOpInternal(RandomDistribution distribution,
                                          absl::Span<const XlaOp> parameters,
                                          const Shape& shape) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_128(mht_128_v, 2852, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngOpInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_distribution(distribution);

  return AddInstruction(std::move(instr), HloOpcode::kRng, parameters);
}

XlaOp XlaBuilder::RngNormal(XlaOp mu, XlaOp sigma, const Shape& shape) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_129(mht_129_v, 2863, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngNormal");

  return RngOp(RandomDistribution::RNG_NORMAL, {mu, sigma}, shape);
}

XlaOp XlaBuilder::RngUniform(XlaOp a, XlaOp b, const Shape& shape) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_130(mht_130_v, 2870, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngUniform");

  return RngOp(RandomDistribution::RNG_UNIFORM, {a, b}, shape);
}

XlaOp XlaBuilder::RngBitGenerator(RandomAlgorithm algorithm,
                                  XlaOp initial_state, const Shape& shape) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_131(mht_131_v, 2878, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngBitGenerator");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
    TF_ASSIGN_OR_RETURN(Shape state_shape, GetShape(initial_state));
    Shape output_shape = shape;
    switch (output_shape.element_type()) {
      case PrimitiveType::F32:
      case PrimitiveType::S32:
      case PrimitiveType::U32:
        output_shape.set_element_type(PrimitiveType::U32);
        break;
      case PrimitiveType::F64:
      case PrimitiveType::S64:
      case PrimitiveType::U64:
        output_shape.set_element_type(PrimitiveType::U64);
        break;
      default:
        return InvalidArgument("Unsupported shape for RngBitGenerator: %s",
                               PrimitiveType_Name(output_shape.element_type()));
    }
    return RngBitGeneratorInternal(
        ShapeUtil::MakeTupleShape({state_shape, output_shape}), algorithm,
        initial_state);
  });
}

StatusOr<XlaOp> XlaBuilder::RngBitGeneratorInternal(
    const Shape& full_result_shape, RandomAlgorithm algorithm,
    XlaOp initial_state) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_132(mht_132_v, 2909, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RngBitGeneratorInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = full_result_shape.ToProto();
  instr.set_rng_algorithm(algorithm);
  return AddInstruction(std::move(instr), HloOpcode::kRngBitGenerator,
                        {initial_state});
}

XlaOp XlaBuilder::While(const XlaComputation& condition,
                        const XlaComputation& body, XlaOp init) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_133(mht_133_v, 2921, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::While");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Infer shape.
    TF_ASSIGN_OR_RETURN(const auto& body_program_shape, body.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const auto& condition_program_shape,
                        condition.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const Shape* init_shape, GetShapePtr(init));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferWhileShape(
                                         condition_program_shape,
                                         body_program_shape, *init_shape));
    return WhileInternal(shape, condition, body, init);
  });
}

StatusOr<XlaOp> XlaBuilder::WhileInternal(const Shape& shape,
                                          const XlaComputation& condition,
                                          const XlaComputation& body,
                                          XlaOp init) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_134(mht_134_v, 2941, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::WhileInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  // Body comes before condition computation in the vector.
  AddCalledComputation(body, &instr);
  AddCalledComputation(condition, &instr);
  return AddInstruction(std::move(instr), HloOpcode::kWhile, {init});
}

XlaOp XlaBuilder::Gather(XlaOp input, XlaOp start_indices,
                         const GatherDimensionNumbers& dimension_numbers,
                         absl::Span<const int64_t> slice_sizes,
                         bool indices_are_sorted) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_135(mht_135_v, 2956, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Gather");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* input_shape, GetShapePtr(input));
    TF_ASSIGN_OR_RETURN(const Shape* start_indices_shape,
                        GetShapePtr(start_indices));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferGatherShape(
                                         *input_shape, *start_indices_shape,
                                         dimension_numbers, slice_sizes));
    return GatherInternal(shape, input, start_indices, dimension_numbers,
                          slice_sizes, indices_are_sorted);
  });
}

StatusOr<XlaOp> XlaBuilder::GatherInternal(
    const Shape& shape, XlaOp input, XlaOp start_indices,
    const GatherDimensionNumbers& dimension_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_136(mht_136_v, 2975, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GatherInternal");

  HloInstructionProto instr;
  instr.set_indices_are_sorted(indices_are_sorted);
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_gather_dimension_numbers() = dimension_numbers;
  for (int64_t bound : slice_sizes) {
    instr.add_gather_slice_sizes(bound);
  }

  return AddInstruction(std::move(instr), HloOpcode::kGather,
                        {input, start_indices});
}

XlaOp XlaBuilder::Scatter(XlaOp input, XlaOp scatter_indices, XlaOp updates,
                          const XlaComputation& update_computation,
                          const ScatterDimensionNumbers& dimension_numbers,
                          bool indices_are_sorted, bool unique_indices) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_137(mht_137_v, 2994, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Scatter");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* input_shape, GetShapePtr(input));
    TF_ASSIGN_OR_RETURN(const Shape* scatter_indices_shape,
                        GetShapePtr(scatter_indices));
    TF_ASSIGN_OR_RETURN(const Shape* updates_shape, GetShapePtr(updates));
    TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                        update_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferScatterShape(
                         *input_shape, *scatter_indices_shape, *updates_shape,
                         to_apply_shape, dimension_numbers));
    return ScatterInternal(shape, input, scatter_indices, updates,
                           update_computation, dimension_numbers,
                           indices_are_sorted, unique_indices);
  });
}

StatusOr<XlaOp> XlaBuilder::ScatterInternal(
    const Shape& shape, XlaOp input, XlaOp scatter_indices, XlaOp updates,
    const XlaComputation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
    bool unique_indices) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_138(mht_138_v, 3019, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ScatterInternal");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    instr.set_indices_are_sorted(indices_are_sorted);
    instr.set_unique_indices(unique_indices);
    *instr.mutable_shape() = shape.ToProto();
    *instr.mutable_scatter_dimension_numbers() = dimension_numbers;

    AddCalledComputation(update_computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kScatter,
                          {input, scatter_indices, updates});
  });
}

XlaOp XlaBuilder::Conditional(XlaOp predicate, XlaOp true_operand,
                              const XlaComputation& true_computation,
                              XlaOp false_operand,
                              const XlaComputation& false_computation) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_139(mht_139_v, 3039, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Conditional");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const xla::Shape* shape, GetShapePtr(predicate));

    if (!ShapeUtil::IsScalar(*shape) || shape->element_type() != PRED) {
      return InvalidArgument(
          "Argument to predicated-Conditional is not a scalar of PRED type "
          "(%s).",
          ShapeUtil::HumanString(*shape));
    }
    // The index of true_computation must be 0 and that of false computation
    // must be 1.
    return ConditionalImpl(predicate, {&true_computation, &false_computation},
                           {true_operand, false_operand});
  });
}

XlaOp XlaBuilder::Conditional(
    XlaOp branch_index,
    absl::Span<const XlaComputation* const> branch_computations,
    absl::Span<const XlaOp> branch_operands) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_140(mht_140_v, 3062, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Conditional");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const xla::Shape* shape, GetShapePtr(branch_index));

    if (!ShapeUtil::IsScalar(*shape) || shape->element_type() != S32) {
      return InvalidArgument(
          "Argument to indexed-Conditional is not a scalar of S32 type (%s).",
          ShapeUtil::HumanString(*shape));
    }
    return ConditionalImpl(branch_index, branch_computations, branch_operands);
  });
}

XlaOp XlaBuilder::ConditionalImpl(
    XlaOp branch_index,
    absl::Span<const XlaComputation* const> branch_computations,
    absl::Span<const XlaOp> branch_operands) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_141(mht_141_v, 3081, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ConditionalImpl");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape* branch_index_shape,
                        GetShapePtr(branch_index));
    std::vector<Shape> branch_operand_shapes(branch_operands.size());
    std::vector<ProgramShape> branch_computation_shapes(
        branch_computations.size());
    for (int j = 0, end = branch_operands.size(); j < end; ++j) {
      TF_ASSIGN_OR_RETURN(branch_operand_shapes[j],
                          GetShape(branch_operands[j]));
      TF_ASSIGN_OR_RETURN(branch_computation_shapes[j],
                          branch_computations[j]->GetProgramShape());
    }
    TF_ASSIGN_OR_RETURN(const Shape shape,
                        ShapeInference::InferConditionalShape(
                            *branch_index_shape, branch_computation_shapes,
                            branch_operand_shapes));
    *instr.mutable_shape() = shape.ToProto();

    for (const XlaComputation* branch_computation : branch_computations) {
      AddCalledComputation(*branch_computation, &instr);
    }

    std::vector<XlaOp> operands(1, branch_index);
    for (const XlaOp branch_operand : branch_operands) {
      operands.emplace_back(branch_operand);
    }
    return AddInstruction(std::move(instr), HloOpcode::kConditional,
                          absl::MakeSpan(operands));
  });
}

Status XlaBuilder::CheckOpBuilder(XlaOp op) const {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_142(mht_142_v, 3118, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CheckOpBuilder");

  if (this != op.builder()) {
    return InvalidArgument(
        "XlaOp with handle %d is built by builder '%s', but is trying to use "
        "it in builder '%s'",
        op.handle(), op.builder()->name(), name());
  }
  return Status::OK();
}

XlaOp XlaBuilder::Reduce(XlaOp operand, XlaOp init_value,
                         const XlaComputation& computation,
                         absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_143(mht_143_v, 3133, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Reduce");

  return Reduce(absl::Span<const XlaOp>({operand}),
                absl::Span<const XlaOp>({init_value}), computation,
                dimensions_to_reduce);
}

XlaOp XlaBuilder::Reduce(absl::Span<const XlaOp> operands,
                         absl::Span<const XlaOp> init_values,
                         const XlaComputation& computation,
                         absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_144(mht_144_v, 3145, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Reduce");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());

    std::vector<XlaOp> all_operands;
    all_operands.insert(all_operands.end(), operands.begin(), operands.end());
    all_operands.insert(all_operands.end(), init_values.begin(),
                        init_values.end());

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes,
                        GetOperandShapes(all_operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });

    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferReduceShape(
            operand_shape_ptrs, dimensions_to_reduce, called_program_shape));
    return ReduceInternal(shape, all_operands, computation,
                          dimensions_to_reduce);
  });
}

StatusOr<XlaOp> XlaBuilder::ReduceInternal(
    const Shape& shape, absl::Span<const XlaOp> all_operands,
    const XlaComputation& computation,
    absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_145(mht_145_v, 3176, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceInternal");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape.ToProto();

    for (int64_t dim : dimensions_to_reduce) {
      instr.add_dimensions(dim);
    }

    AddCalledComputation(computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kReduce, all_operands);
  });
}

XlaOp XlaBuilder::ReduceAll(XlaOp operand, XlaOp init_value,
                            const XlaComputation& computation) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_146(mht_146_v, 3194, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceAll");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<int64_t> all_dimnos(operand_shape->rank());
    std::iota(all_dimnos.begin(), all_dimnos.end(), 0);
    return Reduce(operand, init_value, computation, all_dimnos);
  });
}

XlaOp XlaBuilder::ReduceWindow(XlaOp operand, XlaOp init_value,
                               const XlaComputation& computation,
                               absl::Span<const int64_t> window_dimensions,
                               absl::Span<const int64_t> window_strides,
                               Padding padding) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_147(mht_147_v, 3210, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceWindow");

  return ReduceWindow(absl::MakeSpan(&operand, 1),
                      absl::MakeSpan(&init_value, 1), computation,
                      window_dimensions, window_strides, padding);
}

XlaOp XlaBuilder::ReduceWindow(absl::Span<const XlaOp> operands,
                               absl::Span<const XlaOp> init_values,
                               const XlaComputation& computation,
                               absl::Span<const int64_t> window_dimensions,
                               absl::Span<const int64_t> window_strides,
                               Padding padding) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_148(mht_148_v, 3224, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceWindow");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    const Shape* operand_shape = nullptr;
    for (const auto& operand : operands) {
      TF_ASSIGN_OR_RETURN(operand_shape, GetShapePtr(operand));
      TF_RETURN_IF_ERROR(ValidatePaddingValues(
          operand_shape->dimensions(), window_dimensions, window_strides));
    }
    CHECK(operand_shape != nullptr);
    std::vector<std::pair<int64_t, int64_t>> padding_values =
        MakePadding(operand_shape->dimensions(), window_dimensions,
                    window_strides, padding);
    TF_ASSIGN_OR_RETURN(auto window,
                        ShapeInference::InferWindowFromDimensions(
                            window_dimensions, window_strides, padding_values,
                            /*lhs_dilation=*/{},
                            /*rhs_dilation=*/{}));
    PaddingType padding_type = PADDING_INVALID;
    for (int64_t i = 0; i < operand_shape->rank(); ++i) {
      if (operand_shape->is_dynamic_dimension(i) &&
          !window_util::IsTrivialWindowDimension(window.dimensions(i)) &&
          padding == Padding::kSame) {
        // SAME padding can create dynamic padding sizes. The padding size
        // need to be rewritten by dynamic padder using HloInstructions. We
        // create a CustomCall to handle this.
        padding_type = PADDING_SAME;
      }
    }
    if (padding_type == PADDING_SAME) {
      TF_ASSIGN_OR_RETURN(
          HloInstructionProto instr,
          ReduceWindowInternal(operands, init_values, computation,
                               window_dimensions, window_strides, {}, {},
                               padding_values));
      instr.set_custom_call_target("DynamicReduceWindowSamePadding");
      std::vector<XlaOp> args;
      args.insert(args.end(), operands.begin(), operands.end());
      args.insert(args.end(), init_values.begin(), init_values.end());
      return AddInstruction(std::move(instr), HloOpcode::kCustomCall, args);
    }
    return ReduceWindowWithGeneralPadding(
        operands, init_values, computation, window_dimensions, window_strides,
        /*base_dilations=*/{}, /*window_dilations=*/{}, padding_values);
  });
}

XlaOp XlaBuilder::ReduceWindowWithGeneralPadding(
    absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
    const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_149(mht_149_v, 3280, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceWindowWithGeneralPadding");

  std::vector<const Shape*> operand_shapes, init_shapes;
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (operands.size() == 1) {
      const auto& operand = operands[0];
      const auto& init_value = init_values[0];
      TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
      operand_shapes.push_back(operand_shape);
      TF_ASSIGN_OR_RETURN(const Shape* init_shape, GetShapePtr(init_value));
      init_shapes.push_back(init_shape);

      TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                          computation.GetProgramShape());
      TF_ASSIGN_OR_RETURN(auto window,
                          ShapeInference::InferWindowFromDimensions(
                              window_dimensions, window_strides, padding,
                              /*lhs_dilation=*/base_dilations,
                              /*rhs_dilation=*/window_dilations));
      TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferReduceWindowShape(
                                           absl::MakeSpan(operand_shapes),
                                           absl::MakeSpan(init_shapes), window,
                                           to_apply_shape));
      return ReduceWindowInternal(shape, operands[0], init_values[0],
                                  computation, window);
    }

    TF_ASSIGN_OR_RETURN(
        HloInstructionProto instr,
        ReduceWindowInternal(operands, init_values, computation,
                             window_dimensions, window_strides, base_dilations,
                             window_dilations, padding));
    std::vector<XlaOp> args;
    args.insert(args.end(), operands.begin(), operands.end());
    args.insert(args.end(), init_values.begin(), init_values.end());
    return AddInstruction(std::move(instr), HloOpcode::kReduceWindow, args);
  });
}

StatusOr<HloInstructionProto> XlaBuilder::ReduceWindowInternal(
    absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
    const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_150(mht_150_v, 3328, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceWindowInternal");

  std::vector<const Shape*> operand_shapes, init_shapes;
  for (int i = 0; i < operands.size(); ++i) {
    const auto& operand = operands[i];
    const auto& init_value = init_values[i];
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    operand_shapes.push_back(operand_shape);
    TF_ASSIGN_OR_RETURN(const Shape* init_shape, GetShapePtr(init_value));
    init_shapes.push_back(init_shape);
  }
  TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                      computation.GetProgramShape());
  TF_ASSIGN_OR_RETURN(auto window,
                      ShapeInference::InferWindowFromDimensions(
                          window_dimensions, window_strides, padding,
                          /*lhs_dilation=*/base_dilations,
                          /*rhs_dilation=*/window_dilations));
  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeInference::InferReduceWindowShape(
                          absl::MakeSpan(operand_shapes),
                          absl::MakeSpan(init_shapes), window, to_apply_shape));
  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_window() = std::move(window);
  AddCalledComputation(computation, &instr);
  return instr;
}

StatusOr<XlaOp> XlaBuilder::ReduceWindowInternal(
    const Shape& shape, XlaOp operand, XlaOp init_value,
    const XlaComputation& computation, Window window) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_151(mht_151_v, 3361, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceWindowInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  *instr.mutable_window() = std::move(window);

  AddCalledComputation(computation, &instr);
  return AddInstruction(std::move(instr), HloOpcode::kReduceWindow,
                        {operand, init_value});
}

XlaOp XlaBuilder::BatchNormTraining(XlaOp operand, XlaOp scale, XlaOp offset,
                                    float epsilon, int64_t feature_index) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_152(mht_152_v, 3375, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BatchNormTraining");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* scale_shape, GetShapePtr(scale));
    TF_ASSIGN_OR_RETURN(const Shape* offset_shape, GetShapePtr(offset));
    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferBatchNormTrainingShape(
            *operand_shape, *scale_shape, *offset_shape, feature_index));
    *instr.mutable_shape() = shape.ToProto();

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormTraining,
                          {operand, scale, offset});
  });
}

XlaOp XlaBuilder::BatchNormInference(XlaOp operand, XlaOp scale, XlaOp offset,
                                     XlaOp mean, XlaOp variance, float epsilon,
                                     int64_t feature_index) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_153(mht_153_v, 3401, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BatchNormInference");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* scale_shape, GetShapePtr(scale));
    TF_ASSIGN_OR_RETURN(const Shape* offset_shape, GetShapePtr(offset));
    TF_ASSIGN_OR_RETURN(const Shape* mean_shape, GetShapePtr(mean));
    TF_ASSIGN_OR_RETURN(const Shape* variance_shape, GetShapePtr(variance));
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferBatchNormInferenceShape(
                            *operand_shape, *scale_shape, *offset_shape,
                            *mean_shape, *variance_shape, feature_index));
    *instr.mutable_shape() = shape.ToProto();

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormInference,
                          {operand, scale, offset, mean, variance});
  });
}

XlaOp XlaBuilder::BatchNormGrad(XlaOp operand, XlaOp scale, XlaOp batch_mean,
                                XlaOp batch_var, XlaOp grad_output,
                                float epsilon, int64_t feature_index) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_154(mht_154_v, 3429, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BatchNormGrad");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* scale_shape, GetShapePtr(scale));
    TF_ASSIGN_OR_RETURN(const Shape* batch_mean_shape, GetShapePtr(batch_mean));
    TF_ASSIGN_OR_RETURN(const Shape* batch_var_shape, GetShapePtr(batch_var));
    TF_ASSIGN_OR_RETURN(const Shape* grad_output_shape,
                        GetShapePtr(grad_output));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferBatchNormGradShape(
                         *operand_shape, *scale_shape, *batch_mean_shape,
                         *batch_var_shape, *grad_output_shape, feature_index));
    *instr.mutable_shape() = shape.ToProto();

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormGrad,
                          {operand, scale, batch_mean, batch_var, grad_output});
  });
}

XlaOp XlaBuilder::AllGather(XlaOp operand, int64_t all_gather_dimension,
                            int64_t shard_count,
                            absl::Span<const ReplicaGroup> replica_groups,
                            const absl::optional<ChannelHandle>& channel_id,
                            const absl::optional<Layout>& layout,
                            const absl::optional<bool> use_global_device_ids) {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_155(mht_155_v, 3461, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AllGather");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

    TF_ASSIGN_OR_RETURN(
        Shape inferred_shape,
        ShapeInference::InferAllGatherShape({operand_shape},
                                            all_gather_dimension, shard_count));
    if (layout) {
      *inferred_shape.mutable_layout() = *layout;
      instr.set_constrain_layout(true);
    }
    *instr.mutable_shape() = inferred_shape.ToProto();

    instr.add_dimensions(all_gather_dimension);
    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }
    if (channel_id.has_value()) {
      instr.set_channel_id(channel_id->handle());
    }
    if (use_global_device_ids.has_value()) {
      instr.set_use_global_device_ids(use_global_device_ids.value());
    }

    TF_ASSIGN_OR_RETURN(
        auto all_gather,
        AddInstruction(std::move(instr), HloOpcode::kAllGather, {operand}));
    return all_gather;
  });
}

XlaOp XlaBuilder::CrossReplicaSum(
    XlaOp operand, absl::Span<const ReplicaGroup> replica_groups) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_156(mht_156_v, 3498, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CrossReplicaSum");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    const Shape* element_shape;
    if (shape->IsTuple()) {
      if (shape->tuple_shapes_size() == 0) {
        return Unimplemented(
            "0 element tuple CrossReplicaSum is not supported");
      }
      element_shape = &shape->tuple_shapes(0);
    } else {
      element_shape = shape;
    }
    const Shape scalar_shape =
        ShapeUtil::MakeShape(element_shape->element_type(), {});
    auto b = CreateSubBuilder("sum");
    auto x = b->Parameter(/*parameter_number=*/0, scalar_shape, "x");
    auto y = b->Parameter(/*parameter_number=*/1, scalar_shape, "y");
    if (scalar_shape.element_type() == PRED) {
      Or(x, y);
    } else {
      Add(x, y);
    }
    TF_ASSIGN_OR_RETURN(auto computation, b->Build());
    return AllReduce(operand, computation, replica_groups,
                     /*channel_id=*/absl::nullopt);
  });
}

XlaOp XlaBuilder::AllReduce(XlaOp operand, const XlaComputation& computation,
                            absl::Span<const ReplicaGroup> replica_groups,
                            const absl::optional<ChannelHandle>& channel_id,
                            const absl::optional<Shape>& shape_with_layout) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_157(mht_157_v, 3533, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AllReduce");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> operand_shapes;
    std::vector<XlaOp> operands;
    if (operand_shape->IsTuple()) {
      if (operand_shape->tuple_shapes_size() == 0) {
        return Unimplemented("0 element tuple AllReduce is not supported");
      }
      for (int i = 0; i < operand_shape->tuple_shapes_size(); ++i) {
        if (operand_shape->tuple_shapes(i).element_type() !=
            operand_shape->tuple_shapes(0).element_type()) {
          return Unimplemented(
              "All the shapes of a tuple input of AllReduce must have the same "
              "element type");
        }
        operand_shapes.push_back(&operand_shape->tuple_shapes(i));
        operands.push_back(GetTupleElement(operand, i));
      }
    } else {
      operand_shapes.push_back(operand_shape);
      operands.push_back(operand);
    }

    TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                        ShapeInference::InferAllReduceShape(operand_shapes));
    if (shape_with_layout) {
      if (!LayoutUtil::HasLayout(*shape_with_layout)) {
        return InvalidArgument("shape_with_layout must have the layout set: %s",
                               shape_with_layout->ToString());
      }
      if (!ShapeUtil::Compatible(*shape_with_layout, *operand_shape)) {
        return InvalidArgument(
            "Provided shape_with_layout must be compatible with the "
            "operand shape: %s vs %s",
            shape_with_layout->ToString(), operand_shape->ToString());
      }
      instr.set_constrain_layout(true);
      if (operand_shape->IsTuple() && !inferred_shape.IsTuple()) {
        // For a single-element tuple, take the tuple element shape.
        TF_RET_CHECK(shape_with_layout->tuple_shapes_size() == 1);
        *instr.mutable_shape() = shape_with_layout->tuple_shapes(0).ToProto();
      } else {
        *instr.mutable_shape() = shape_with_layout->ToProto();
      }
    } else {
      *instr.mutable_shape() = inferred_shape.ToProto();
    }

    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }

    if (channel_id.has_value()) {
      instr.set_channel_id(channel_id->handle());
    }

    AddCalledComputation(computation, &instr);

    TF_ASSIGN_OR_RETURN(
        auto all_reduce,
        AddInstruction(std::move(instr), HloOpcode::kAllReduce, operands));
    if (operand_shape->IsTuple() && !inferred_shape.IsTuple()) {
      // For a single-element tuple, wrap the result into a tuple.
      TF_RET_CHECK(operand_shapes.size() == 1);
      TF_RET_CHECK(ShapeUtil::Compatible(*operand_shapes[0], inferred_shape));
      return Tuple({all_reduce});
    }
    return all_reduce;
  });
}

XlaOp XlaBuilder::ReduceScatter(
    XlaOp operand, const XlaComputation& computation, int64_t scatter_dimension,
    int64_t shard_count, absl::Span<const ReplicaGroup> replica_groups,
    const absl::optional<ChannelHandle>& channel_id,
    const absl::optional<Layout>& layout,
    const absl::optional<bool> use_global_device_ids) {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_158(mht_158_v, 3614, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReduceScatter");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    std::vector<const Shape*> operand_shapes;
    std::vector<XlaOp> operands;
    if (operand_shape->IsTuple()) {
      if (operand_shape->tuple_shapes_size() == 0) {
        return Unimplemented("0 element tuple ReduceScatter is not supported");
      }
      for (int i = 0; i < operand_shape->tuple_shapes_size(); ++i) {
        if (operand_shape->tuple_shapes(i).element_type() !=
            operand_shape->tuple_shapes(0).element_type()) {
          return Unimplemented(
              "All the shapes of a tuple input of ReduceScatter must have "
              "the same "
              "element type");
        }
        operand_shapes.push_back(&operand_shape->tuple_shapes(i));
        operands.push_back(GetTupleElement(operand, i));
      }
    } else {
      operand_shapes.push_back(operand_shape);
      operands.push_back(operand);
    }

    TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                        ShapeInference::InferReduceScatterShape(
                            operand_shapes, scatter_dimension, shard_count));
    if (layout) {
      *inferred_shape.mutable_layout() = *layout;
      instr.set_constrain_layout(true);
    }
    *instr.mutable_shape() = inferred_shape.ToProto();

    AddCalledComputation(computation, &instr);

    instr.add_dimensions(scatter_dimension);
    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }
    if (channel_id.has_value()) {
      instr.set_channel_id(channel_id->handle());
    }
    if (use_global_device_ids.has_value()) {
      instr.set_use_global_device_ids(use_global_device_ids.value());
    }

    TF_ASSIGN_OR_RETURN(
        auto reduce_scatter,
        AddInstruction(std::move(instr), HloOpcode::kReduceScatter, {operand}));
    return reduce_scatter;
  });
}

XlaOp XlaBuilder::AllToAll(XlaOp operand, int64_t split_dimension,
                           int64_t concat_dimension, int64_t split_count,
                           absl::Span<const ReplicaGroup> replica_groups,
                           const absl::optional<Layout>& layout) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_159(mht_159_v, 3675, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AllToAll");

  // Array all_to_all may need to violate layout constraint to be legal so use
  // the tuple version.
  if (layout.has_value()) {
    return AllToAllTuple(operand, split_dimension, concat_dimension,
                         split_count, replica_groups, layout);
  }
  return AllToAllArray(operand, split_dimension, concat_dimension, split_count,
                       replica_groups);
}

XlaOp XlaBuilder::AllToAllArray(XlaOp operand, int64_t split_dimension,
                                int64_t concat_dimension, int64_t split_count,
                                absl::Span<const ReplicaGroup> replica_groups) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_160(mht_160_v, 3691, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AllToAllArray");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        const Shape all_to_all_shape,
        ShapeInference::InferAllToAllShape(*operand_shape, split_dimension,
                                           concat_dimension, split_count));
    HloInstructionProto instr;
    *instr.mutable_shape() = operand_shape->ToProto();
    if (replica_groups.empty()) {
      auto* group = instr.add_replica_groups();
      for (int64_t i = 0; i < split_count; ++i) {
        group->add_replica_ids(i);
      }
    } else {
      for (const ReplicaGroup& group : replica_groups) {
        *instr.add_replica_groups() = group;
      }
    }
    instr.add_dimensions(split_dimension);
    TF_ASSIGN_OR_RETURN(
        XlaOp all_to_all,
        AddInstruction(std::move(instr), HloOpcode::kAllToAll, {operand}));
    if (split_dimension == concat_dimension) {
      return all_to_all;
    }
    DimensionVector sizes;
    for (int64_t i = 0; i < operand_shape->rank(); ++i) {
      if (i != split_dimension) {
        sizes.push_back(operand_shape->dimensions(i));
        continue;
      }
      sizes.push_back(split_count);
      sizes.push_back(operand_shape->dimensions(i) / split_count);
    }
    all_to_all = Reshape(all_to_all, sizes);

    std::vector<int64_t> permutation;
    const auto rank = operand_shape->rank();
    permutation.reserve(rank + 1);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t dim_after_reshape = i >= split_dimension ? i + 1 : i;
      if (i == concat_dimension) {
        permutation.push_back(split_dimension);
      }
      permutation.push_back(dim_after_reshape);
    }
    all_to_all = Transpose(all_to_all, permutation);
    return Reshape(all_to_all_shape, all_to_all);
  });
}

XlaOp XlaBuilder::AllToAllTuple(XlaOp operand, int64_t split_dimension,
                                int64_t concat_dimension, int64_t split_count,
                                absl::Span<const ReplicaGroup> replica_groups,
                                const absl::optional<Layout>& layout) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_161(mht_161_v, 3749, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AllToAllTuple");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

    // The HloInstruction for Alltoall currently only handles the data
    // communication: it accepts N already split parts and scatters them to N
    // cores, and each core gathers the N received parts into a tuple as the
    // output. So here we explicitly split the operand before the hlo alltoall,
    // and concat the tuple elements.
    //
    // First, run shape inference to make sure the shapes are valid.
    TF_RETURN_IF_ERROR(
        ShapeInference::InferAllToAllShape(*operand_shape, split_dimension,
                                           concat_dimension, split_count)
            .status());

    // Split into N parts.
    std::vector<XlaOp> slices;
    slices.reserve(split_count);
    const int64_t block_size =
        operand_shape->dimensions(split_dimension) / split_count;
    for (int i = 0; i < split_count; i++) {
      slices.push_back(SliceInDim(operand, /*start_index=*/i * block_size,
                                  /*limit_index=*/(i + 1) * block_size,
                                  /*stride=*/1, /*dimno=*/split_dimension));
    }

    // Handle data communication.
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(auto slice_shapes, this->GetOperandShapes(slices));
    std::vector<const Shape*> slice_shape_ptrs;
    absl::c_transform(slice_shapes, std::back_inserter(slice_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferAllToAllTupleShape(slice_shape_ptrs));

    if (layout) {
      TF_RET_CHECK(shape.IsTuple() && !ShapeUtil::IsNestedTuple(shape));
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
        const int64_t layout_minor_to_major_size =
            layout->minor_to_major().size();
        if (layout_minor_to_major_size != shape.tuple_shapes(i).rank()) {
          return InvalidArgument(
              "Provided layout must be compatible with the operand shape: %s "
              "vs %s",
              layout->ToString(), operand_shape->ToString());
        }
        *(shape.mutable_tuple_shapes(i)->mutable_layout()) = *layout;
      }
      instr.set_constrain_layout(true);
    }
    *instr.mutable_shape() = shape.ToProto();

    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }
    TF_ASSIGN_OR_RETURN(
        XlaOp alltoall,
        AddInstruction(std::move(instr), HloOpcode::kAllToAll, slices));

    // Concat the N received parts.
    std::vector<XlaOp> received;
    received.reserve(split_count);
    for (int i = 0; i < split_count; i++) {
      received.push_back(this->GetTupleElement(alltoall, i));
    }
    return this->ConcatInDim(received, concat_dimension);
  });
}

XlaOp XlaBuilder::CollectivePermute(
    XlaOp operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_162(mht_162_v, 3824, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CollectivePermute");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(
        Shape shape,
        ShapeInference::InferCollectivePermuteShape({operand_shape}));
    *instr.mutable_shape() = shape.ToProto();

    for (const auto& pair : source_target_pairs) {
      auto* proto_pair = instr.add_source_target_pairs();
      proto_pair->set_source(pair.first);
      proto_pair->set_target(pair.second);
    }

    return AddInstruction(std::move(instr), HloOpcode::kCollectivePermute,
                          {operand});
  });
}

XlaOp XlaBuilder::ReplicaId() {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_163(mht_163_v, 3847, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReplicaId");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeShape(U32, {}).ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kReplicaId, {});
  });
}

XlaOp XlaBuilder::SelectAndScatter(XlaOp operand, const XlaComputation& select,
                                   absl::Span<const int64_t> window_dimensions,
                                   absl::Span<const int64_t> window_strides,
                                   Padding padding, XlaOp source,
                                   XlaOp init_value,
                                   const XlaComputation& scatter) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_164(mht_164_v, 3863, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SelectAndScatter");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

    std::vector<std::pair<int64_t, int64_t>> padding_values =
        MakePadding(operand_shape->dimensions(), window_dimensions,
                    window_strides, padding);

    TF_ASSIGN_OR_RETURN(auto window,
                        ShapeInference::InferWindowFromDimensions(
                            window_dimensions, window_strides, padding_values,
                            /*lhs_dilation=*/{},
                            /*rhs_dilation=*/{}));
    PaddingType padding_type = PADDING_INVALID;
    for (int64_t i = 0; i < operand_shape->rank(); ++i) {
      if (operand_shape->is_dynamic_dimension(i) &&
          !window_util::IsTrivialWindowDimension(window.dimensions(i)) &&
          padding == Padding::kSame) {
        // SAME padding can create dynamic padding sizes. The padding size
        // need to be rewritten by dynamic padder using HloInstructions. We
        // create a CustomCall to handle this.
        padding_type = PADDING_SAME;
      }
    }
    if (padding_type == PADDING_SAME) {
      TF_ASSIGN_OR_RETURN(
          HloInstructionProto instr,
          SelectAndScatterInternal(operand, select, window_dimensions,
                                   window_strides, padding_values, source,
                                   init_value, scatter));
      instr.set_custom_call_target("DynamicSelectAndScatterSamePadding");
      return AddInstruction(std::move(instr), HloOpcode::kCustomCall,
                            {operand, source, init_value});
    }
    return SelectAndScatterWithGeneralPadding(
        operand, select, window_dimensions, window_strides, padding_values,
        source, init_value, scatter);
  });
}

StatusOr<HloInstructionProto> XlaBuilder::SelectAndScatterInternal(
    XlaOp operand, const XlaComputation& select,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
    XlaOp init_value, const XlaComputation& scatter) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_165(mht_165_v, 3911, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SelectAndScatterInternal");

  HloInstructionProto instr;

  TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
  TF_ASSIGN_OR_RETURN(const Shape* source_shape, GetShapePtr(source));
  TF_ASSIGN_OR_RETURN(const Shape* init_shape, GetShapePtr(init_value));
  TF_ASSIGN_OR_RETURN(const ProgramShape& select_shape,
                      select.GetProgramShape());
  TF_ASSIGN_OR_RETURN(const ProgramShape& scatter_shape,
                      scatter.GetProgramShape());
  TF_ASSIGN_OR_RETURN(*instr.mutable_window(),
                      ShapeInference::InferWindowFromDimensions(
                          window_dimensions, window_strides, padding,
                          /*lhs_dilation=*/{}, /*rhs_dilation=*/{}));
  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeInference::InferSelectAndScatterShape(
                          *operand_shape, select_shape, instr.window(),
                          *source_shape, *init_shape, scatter_shape));
  *instr.mutable_shape() = shape.ToProto();

  AddCalledComputation(select, &instr);
  AddCalledComputation(scatter, &instr);
  return instr;
}

XlaOp XlaBuilder::SelectAndScatterWithGeneralPadding(
    XlaOp operand, const XlaComputation& select,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
    XlaOp init_value, const XlaComputation& scatter) {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_166(mht_166_v, 3944, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SelectAndScatterWithGeneralPadding");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(HloInstructionProto instr,
                        SelectAndScatterInternal(
                            operand, select, window_dimensions, window_strides,
                            padding, source, init_value, scatter));

    return AddInstruction(std::move(instr), HloOpcode::kSelectAndScatter,
                          {operand, source, init_value});
  });
}

XlaOp XlaBuilder::ReducePrecision(XlaOp operand, const int exponent_bits,
                                  const int mantissa_bits) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_167(mht_167_v, 3960, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReducePrecision");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferReducePrecisionShape(
                            *operand_shape, exponent_bits, mantissa_bits));
    return ReducePrecisionInternal(shape, operand, exponent_bits,
                                   mantissa_bits);
  });
}

StatusOr<XlaOp> XlaBuilder::ReducePrecisionInternal(const Shape& shape,
                                                    XlaOp operand,
                                                    const int exponent_bits,
                                                    const int mantissa_bits) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_168(mht_168_v, 3977, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::ReducePrecisionInternal");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.set_exponent_bits(exponent_bits);
  instr.set_mantissa_bits(mantissa_bits);
  return AddInstruction(std::move(instr), HloOpcode::kReducePrecision,
                        {operand});
}

void XlaBuilder::Send(XlaOp operand, const ChannelHandle& handle) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_169(mht_169_v, 3989, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Send");

  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Send HLO takes two operands: a data operand and a token. Generate the
    // token to pass into the send.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    return SendWithToken(operand, token, handle);
  });
}

XlaOp XlaBuilder::SendWithToken(XlaOp operand, XlaOp token,
                                const ChannelHandle& handle) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_170(mht_170_v, 4008, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SendWithToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (handle.type() != ChannelHandle::DEVICE_TO_DEVICE) {
      return InvalidArgument("Send must use a device-to-device channel");
    }

    // Send instruction produces a tuple of {aliased operand, U32 context,
    // token}.
    HloInstructionProto send_instr;
    TF_ASSIGN_OR_RETURN(const Shape* shape, GetShapePtr(operand));
    *send_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({*shape, ShapeUtil::MakeShape(U32, {}),
                                   ShapeUtil::MakeTokenShape()})
            .ToProto();
    send_instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(XlaOp send,
                        AddInstruction(std::move(send_instr), HloOpcode::kSend,
                                       {operand, token}));

    HloInstructionProto send_done_instr;
    *send_done_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    send_done_instr.set_channel_id(handle.handle());
    return AddInstruction(std::move(send_done_instr), HloOpcode::kSendDone,
                          {send});
  });
}

XlaOp XlaBuilder::Recv(const Shape& shape, const ChannelHandle& handle) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_171(mht_171_v, 4038, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Recv");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Recv HLO takes a single token operand. Generate the token to pass into
    // the Recv and RecvDone instructions.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    XlaOp recv = RecvWithToken(token, shape, handle);

    // The RecvDone instruction produces a tuple of the data and a token
    // type. Return XLA op containing the data.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto recv_data;
    *recv_data.mutable_shape() = shape.ToProto();
    recv_data.set_tuple_index(0);
    return AddInstruction(std::move(recv_data), HloOpcode::kGetTupleElement,
                          {recv});
  });
}

XlaOp XlaBuilder::RecvWithToken(XlaOp token, const Shape& shape,
                                const ChannelHandle& handle) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_172(mht_172_v, 4067, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RecvWithToken");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (handle.type() != ChannelHandle::DEVICE_TO_DEVICE) {
      return InvalidArgument("Recv must use a device-to-device channel");
    }

    // Recv instruction produces a tuple of {receive buffer, U32 context,
    // token}.
    HloInstructionProto recv_instr;
    *recv_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape(
            {shape, ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()})
            .ToProto();
    recv_instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(XlaOp recv, AddInstruction(std::move(recv_instr),
                                                   HloOpcode::kRecv, {token}));

    HloInstructionProto recv_done_instr;
    *recv_done_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()})
            .ToProto();
    recv_done_instr.set_channel_id(handle.handle());
    return AddInstruction(std::move(recv_done_instr), HloOpcode::kRecvDone,
                          {recv});
  });
}

XlaOp XlaBuilder::SendToHost(XlaOp operand, XlaOp token,
                             const Shape& shape_with_layout,
                             const ChannelHandle& handle) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_173(mht_173_v, 4099, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SendToHost");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Shape passed to SendToHost must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    if (!ShapeUtil::Compatible(*operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "SendToHost shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(*operand_shape));
    }
    // TODO(b/111544877): Support tuple shapes.
    if (!operand_shape->IsArray()) {
      return InvalidArgument("SendToHost only supports array shapes, shape: %s",
                             ShapeUtil::HumanString(*operand_shape));
    }

    if (handle.type() != ChannelHandle::DEVICE_TO_HOST) {
      return InvalidArgument("SendToHost must use a device-to-host channel");
    }

    // Send instruction produces a tuple of {aliased operand, U32 context,
    // token}.
    HloInstructionProto send_instr;
    *send_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape_with_layout,
                                   ShapeUtil::MakeShape(U32, {}),
                                   ShapeUtil::MakeTokenShape()})
            .ToProto();
    send_instr.set_channel_id(handle.handle());
    send_instr.set_is_host_transfer(true);
    TF_ASSIGN_OR_RETURN(XlaOp send,
                        AddInstruction(std::move(send_instr), HloOpcode::kSend,
                                       {operand, token}));

    HloInstructionProto send_done_instr;
    *send_done_instr.mutable_shape() = ShapeUtil::MakeTokenShape().ToProto();
    send_done_instr.set_channel_id(handle.handle());
    send_done_instr.set_is_host_transfer(true);
    return AddInstruction(std::move(send_done_instr), HloOpcode::kSendDone,
                          {send});
  });
}

XlaOp XlaBuilder::RecvFromHost(XlaOp token, const Shape& shape,
                               const ChannelHandle& handle) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_174(mht_174_v, 4148, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RecvFromHost");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Shape passed to RecvFromHost must have a layout");
    }

    // TODO(b/111544877): Support tuple shapes.
    if (!shape.IsArray()) {
      return InvalidArgument(
          "RecvFromHost only supports array shapes, shape: %s",
          ShapeUtil::HumanString(shape));
    }

    if (handle.type() != ChannelHandle::HOST_TO_DEVICE) {
      return InvalidArgument("RecvFromHost must use a host-to-device channel");
    }

    // Recv instruction produces a tuple of {receive buffer, U32 context,
    // token}.
    HloInstructionProto recv_instr;
    *recv_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape(
            {shape, ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()})
            .ToProto();
    recv_instr.set_channel_id(handle.handle());
    recv_instr.set_is_host_transfer(true);
    TF_ASSIGN_OR_RETURN(XlaOp recv, AddInstruction(std::move(recv_instr),
                                                   HloOpcode::kRecv, {token}));

    HloInstructionProto recv_done_instr;
    *recv_done_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()})
            .ToProto();
    recv_done_instr.set_channel_id(handle.handle());
    recv_done_instr.set_is_host_transfer(true);
    return AddInstruction(std::move(recv_done_instr), HloOpcode::kRecvDone,
                          {recv});
  });
}

XlaOp XlaBuilder::GetDimensionSize(XlaOp operand, int64_t dimension) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_175(mht_175_v, 4191, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::GetDimensionSize");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferGetDimensionSizeShape(
                                         *operand_shape, dimension));
    // Calling GetDimensionSize on a static dimension returns a constant
    // instruction.
    if (!operand_shape->is_dynamic_dimension(dimension)) {
      return ConstantR0<int32_t>(this, operand_shape->dimensions(dimension));
    }
    *instr.mutable_shape() = shape.ToProto();
    instr.add_dimensions(dimension);
    return AddInstruction(std::move(instr), HloOpcode::kGetDimensionSize,
                          {operand});
  });
}

XlaOp XlaBuilder::RemoveDynamicDimension(XlaOp operand, int64_t dimension) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_176(mht_176_v, 4212, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::RemoveDynamicDimension");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));

    Shape shape = *operand_shape;
    shape.set_dynamic_dimension(dimension, false);
    // Setting an op's dynamic dimension to its static size removes the dynamic
    // dimension.
    XlaOp static_size =
        ConstantR0<int32_t>(this, operand_shape->dimensions(dimension));
    return SetDimensionSizeInternal(shape, operand, static_size, dimension);
  });
}

XlaOp XlaBuilder::SetDimensionSize(XlaOp operand, XlaOp val,
                                   int64_t dimension) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_177(mht_177_v, 4231, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SetDimensionSize");

  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(const Shape* val_shape, GetShapePtr(val));

    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferSetDimensionSizeShape(
                            *operand_shape, *val_shape, dimension));
    return SetDimensionSizeInternal(shape, operand, val, dimension);
  });
}

StatusOr<XlaOp> XlaBuilder::SetDimensionSizeInternal(const Shape& shape,
                                                     XlaOp operand, XlaOp val,
                                                     int64_t dimension) {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_178(mht_178_v, 4248, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::SetDimensionSizeInternal");

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* val_proto,
                      LookUpInstruction(val));
  if (StringToHloOpcode(val_proto->opcode()).ValueOrDie() ==
          HloOpcode::kConstant &&
      shape.is_dynamic_dimension(dimension)) {
    TF_ASSIGN_OR_RETURN(auto constant_size,
                        Literal::CreateFromProto(val_proto->literal(), true));
    if (constant_size.Get<int32_t>({}) == shape.dimensions(dimension)) {
      return operand;
    }
  }

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  instr.add_dimensions(dimension);
  return AddInstruction(std::move(instr), HloOpcode::kSetDimensionSize,
                        {operand, val});
}

StatusOr<bool> XlaBuilder::IsConstant(XlaOp operand) const {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_179(mht_179_v, 4271, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::IsConstant");

  TF_RETURN_IF_ERROR(first_error_);

  // Verify that the handle is valid.
  TF_RETURN_IF_ERROR(LookUpInstruction(operand).status());

  bool is_constant = true;
  absl::flat_hash_set<int64_t> visited;
  IsConstantVisitor(operand.handle(), /*depth=*/0, &visited, &is_constant);
  return is_constant;
}

StatusOr<XlaComputation> XlaBuilder::BuildConstantSubGraph(
    XlaOp root_op, bool dynamic_dimension_is_minus_one) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_180(mht_180_v, 4287, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::BuildConstantSubGraph");

  TF_ASSIGN_OR_RETURN(bool is_constant, IsConstant(root_op));
  if (!is_constant) {
    auto op_status = LookUpInstruction(root_op);
    std::string op_string =
        op_status.ok() ? op_status.ValueOrDie()->name() : "<unknown operation>";
    return InvalidArgument(
        "Operand to BuildConstantSubGraph depends on a parameter.\n\n"
        "  op requested for constant subgraph: %s\n\n"
        "This is an internal error that typically happens when the XLA user "
        "(e.g. TensorFlow) is attempting to determine a value that must be a "
        "compile-time constant (e.g. an array dimension) but it is not capable "
        "of being evaluated at XLA compile time.\n\n"
        "Please file a usability bug with the framework being used (e.g. "
        "TensorFlow).",
        op_string);
  }

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      LookUpInstruction(root_op));
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Build constant subgraph for:\n" << OpToString(root_op);
  }

  HloComputationProto entry;
  SetProtoIdAndName(&entry, StrCat(name_, "_compute_constant"), kNameSeparator,
                    GetNextId());
  ProgramShapeProto* program_shape = entry.mutable_program_shape();
  *program_shape->mutable_result() = root->shape();

  // We use std::set to keep the instruction ids in ascending order (which is
  // also a valid dependency order). The related ops will be added to the
  // subgraph in the same order.
  std::set<int64_t> related_ops;
  absl::flat_hash_map<int64_t, int64_t> substitutions;
  absl::flat_hash_set<int64_t> related_calls;  // Related computations.
  std::queue<int64_t> worklist;
  worklist.push(root->id());
  related_ops.insert(root->id());

  while (!worklist.empty()) {
    int64_t handle = worklist.front();
    worklist.pop();
    TF_ASSIGN_OR_RETURN(const HloInstructionProto* instr_proto,
                        LookUpInstructionByHandle(handle));

    auto default_behavior = [&related_ops, &worklist, &related_calls,
                             instr_proto]() {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_181(mht_181_v, 4337, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "lambda");

      for (int64_t id : instr_proto->operand_ids()) {
        if (related_ops.insert(id).second) {
          worklist.push(id);
        }
      }
      for (int64_t called_id : instr_proto->called_computation_ids()) {
        related_calls.insert(called_id);
      }
    };

    if (instr_proto->opcode() ==
            HloOpcodeString(HloOpcode::kGetDimensionSize) ||
        InstrIsSetBound(instr_proto)) {
      int32_t constant_value = -1;
      HloInstructionProto const_instr;

      if (instr_proto->opcode() ==
          HloOpcodeString(HloOpcode::kGetDimensionSize)) {
        // At this point, BuildConstantSubGraph should never encounter a
        // GetDimensionSize with a dynamic dimension. IsConstant check would
        // have failed at the beginning of this function.
        //
        // Replace GetDimensionSize with a Constant representing the static
        // bound of the shape.
        int64_t dimension = instr_proto->dimensions(0);
        int64_t operand_handle = instr_proto->operand_ids(0);
        TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                            LookUpInstructionByHandle(operand_handle));

        if (!(operand_proto->shape().is_dynamic_dimension(dimension) &&
              dynamic_dimension_is_minus_one)) {
          constant_value = static_cast<int32_t>(
              operand_proto->shape().dimensions(dimension));
        }
        Literal literal = LiteralUtil::CreateR0(constant_value);
        *const_instr.mutable_literal() = literal.ToProto();
        *const_instr.mutable_shape() = literal.shape().ToProto();
      } else {
        if (instr_proto->literal().shape().element_type() == TUPLE) {
          *const_instr.mutable_literal() =
              // First literal of SetBound contains bounds, second literal
              // contains dynamism indicators.
              instr_proto->literal().tuple_literals(0);
        } else {
          *const_instr.mutable_literal() = instr_proto->literal();
        }

        *const_instr.mutable_shape() = instr_proto->shape();
      }
      *const_instr.mutable_opcode() = HloOpcodeString(HloOpcode::kConstant);
      const_instr.set_id(handle);
      *const_instr.mutable_name() =
          GetFullName(const_instr.opcode(), kNameSeparator, const_instr.id());
      *entry.add_instructions() =
          const_instr;  // Add to the result constant graph.

    } else if (instr_proto->opcode() ==
               HloOpcodeString(HloOpcode::kGetTupleElement)) {
      // Look through GTE(Tuple(..), i).
      TF_ASSIGN_OR_RETURN(
          const HloInstructionProto* maybe_tuple_instr,
          LookUpInstructionByHandle(instr_proto->operand_ids(0)));

      if (maybe_tuple_instr->opcode() == HloOpcodeString(HloOpcode::kTuple)) {
        int64_t id = maybe_tuple_instr->operand_ids(instr_proto->tuple_index());
        // Enqueue any dependencies of `id`.
        if (related_ops.insert(id).second) {
          worklist.push(id);
        }
        substitutions[handle] = id;

      } else {
        default_behavior();
      }

    } else {
      default_behavior();
    }
  }

  // Resolve any substitutions for the root id.
  int64_t root_id = root->id();
  auto it = substitutions.find(root_id);
  while (it != substitutions.end()) {
    root_id = it->second;
    it = substitutions.find(root_id);
  }
  entry.set_root_id(root_id);

  // Add related ops to the computation.
  for (int64_t id : related_ops) {
    if (substitutions.find(id) != substitutions.end()) {
      // Skip adding this instruction; we will replace references to it with the
      // substitution instruction's id.
      continue;
    }
    TF_ASSIGN_OR_RETURN(const HloInstructionProto* instr_src,
                        LookUpInstructionByHandle(id));

    if (instr_src->opcode() == HloOpcodeString(HloOpcode::kGetDimensionSize) ||
        InstrIsSetBound(instr_src)) {
      continue;
    }
    HloInstructionProto* instr = entry.add_instructions();
    *instr = *instr_src;
    // Replace operands in case we have substitutions mapped.
    instr->clear_operand_ids();
    for (int64_t operand_id : instr_src->operand_ids()) {
      auto it = substitutions.find(operand_id);
      while (it != substitutions.end()) {
        operand_id = it->second;
        it = substitutions.find(operand_id);
      }
      instr->add_operand_ids(operand_id);
    }
    // Ensures that the instruction names are unique among the graph.
    const std::string& new_name =
        StrCat(instr->name(), ".", entry.id(), ".", instr->id());
    instr->set_name(new_name);
  }

  XlaComputation computation(entry.id());
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
  *module->mutable_host_program_shape() = *program_shape;
  for (auto& e : embedded_) {
    if (related_calls.find(e.second.id()) != related_calls.end()) {
      *module->add_computations() = e.second;
    }
  }
  *module->add_computations() = std::move(entry);
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Constant computation:\n" << module->DebugString();
  }
  return std::move(computation);
}

std::unique_ptr<XlaBuilder> XlaBuilder::CreateSubBuilder(
    const std::string& computation_name) {
   std::vector<std::string> mht_182_v;
   mht_182_v.push_back("computation_name: \"" + computation_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_182(mht_182_v, 4483, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CreateSubBuilder");

  auto sub_builder = absl::make_unique<XlaBuilder>(computation_name);
  sub_builder->parent_builder_ = this;
  sub_builder->die_immediately_on_error_ = this->die_immediately_on_error_;
  return sub_builder;
}

/* static */ ConvolutionDimensionNumbers
XlaBuilder::CreateDefaultConvDimensionNumbers(int num_spatial_dims) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_183(mht_183_v, 4494, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::CreateDefaultConvDimensionNumbers");

  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_input_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_output_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_output_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_kernel_output_feature_dimension(
      kConvKernelOutputDimension);
  dimension_numbers.set_kernel_input_feature_dimension(
      kConvKernelInputDimension);
  for (int i = 0; i < num_spatial_dims; ++i) {
    dimension_numbers.add_input_spatial_dimensions(i + 2);
    dimension_numbers.add_kernel_spatial_dimensions(i + 2);
    dimension_numbers.add_output_spatial_dimensions(i + 2);
  }
  return dimension_numbers;
}

/* static */ Status XlaBuilder::Validate(
    const ConvolutionDimensionNumbers& dnum) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_184(mht_184_v, 4516, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::Validate");

  if (dnum.input_spatial_dimensions_size() < 2) {
    return FailedPrecondition("input spacial dimension < 2: %d",
                              dnum.input_spatial_dimensions_size());
  }
  if (dnum.kernel_spatial_dimensions_size() < 2) {
    return FailedPrecondition("kernel spacial dimension < 2: %d",
                              dnum.kernel_spatial_dimensions_size());
  }
  if (dnum.output_spatial_dimensions_size() < 2) {
    return FailedPrecondition("output spacial dimension < 2: %d",
                              dnum.output_spatial_dimensions_size());
  }

  if (std::set<int64_t>(
          {dnum.input_batch_dimension(), dnum.input_feature_dimension(),
           dnum.input_spatial_dimensions(0), dnum.input_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the input are not unique: (%d, %d, %d, "
        "%d)",
        dnum.input_batch_dimension(), dnum.input_feature_dimension(),
        dnum.input_spatial_dimensions(0), dnum.input_spatial_dimensions(1));
  }
  if (std::set<int64_t>({dnum.kernel_output_feature_dimension(),
                         dnum.kernel_input_feature_dimension(),
                         dnum.kernel_spatial_dimensions(0),
                         dnum.kernel_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the weight are not unique: (%d, %d, %d, "
        "%d)",
        dnum.kernel_output_feature_dimension(),
        dnum.kernel_input_feature_dimension(),
        dnum.kernel_spatial_dimensions(0), dnum.kernel_spatial_dimensions(1));
  }
  if (std::set<int64_t>({dnum.output_batch_dimension(),
                         dnum.output_feature_dimension(),
                         dnum.output_spatial_dimensions(0),
                         dnum.output_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the output are not unique: (%d, %d, %d, "
        "%d)",
        dnum.output_batch_dimension(), dnum.output_feature_dimension(),
        dnum.output_spatial_dimensions(0), dnum.output_spatial_dimensions(1));
  }
  return Status::OK();
}

StatusOr<XlaOp> XlaBuilder::AddInstruction(HloInstructionProto&& instr,
                                           HloOpcode opcode,
                                           absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_185(mht_185_v, 4571, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AddInstruction");

  TF_RETURN_IF_ERROR(first_error_);

  const int64_t handle = GetNextId();
  instr.set_id(handle);
  instr.set_opcode(HloOpcodeString(opcode));
  if (instr.name().empty()) {
    instr.set_name(instr.opcode());
  }
  for (const auto& operand : operands) {
    if (operand.builder_ == nullptr) {
      return InvalidArgument("invalid XlaOp with handle %d", operand.handle());
    }
    if (operand.builder_ != this) {
      return InvalidArgument("Do not add XlaOp from builder %s to builder %s",
                             operand.builder_->name(), this->name());
    }
    instr.add_operand_ids(operand.handle());
  }

  if (one_shot_metadata_.has_value()) {
    *instr.mutable_metadata() = one_shot_metadata_.value();
    one_shot_metadata_.reset();
  } else {
    *instr.mutable_metadata() = metadata_;
  }
  if (sharding_) {
    *instr.mutable_sharding() = *sharding_;
  }
  *instr.mutable_frontend_attributes() = frontend_attributes_;

  handle_to_index_[handle] = instructions_.size();
  instructions_.push_back(std::move(instr));
  instruction_shapes_.push_back(
      absl::make_unique<Shape>(instructions_.back().shape()));

  XlaOp op(handle, this);
  return op;
}

StatusOr<XlaOp> XlaBuilder::AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                           absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_186(mht_186_v, 4615, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AddOpWithShape");

  HloInstructionProto instr;
  *instr.mutable_shape() = shape.ToProto();
  return AddInstruction(std::move(instr), opcode, operands);
}

void XlaBuilder::AddCalledComputation(const XlaComputation& computation,
                                      HloInstructionProto* instr) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_187(mht_187_v, 4625, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::AddCalledComputation");

  absl::flat_hash_map<int64_t, int64_t> remapped_ids;
  std::vector<HloComputationProto> imported_computations;
  imported_computations.reserve(computation.proto().computations_size());
  // Before we import the computations by remapping IDs, and capturing the
  // old->new mappings in remapped_ids.
  for (const HloComputationProto& e : computation.proto().computations()) {
    HloComputationProto new_computation(e);
    int64_t computation_id = GetNextId();
    remapped_ids[new_computation.id()] = computation_id;
    SetProtoIdAndName(&new_computation,
                      GetBaseName(new_computation.name(), kNameSeparator),
                      kNameSeparator, computation_id);
    for (auto& instruction : *new_computation.mutable_instructions()) {
      int64_t instruction_id = GetNextId();
      remapped_ids[instruction.id()] = instruction_id;
      SetProtoIdAndName(&instruction,
                        GetBaseName(instruction.name(), kNameSeparator),
                        kNameSeparator, instruction_id);
    }
    new_computation.set_root_id(remapped_ids.at(new_computation.root_id()));

    imported_computations.push_back(std::move(new_computation));
  }
  // Once we have imported all the computations, and captured all the ID
  // mappings, we go back and fixup the IDs in the imported computations.
  instr->add_called_computation_ids(
      remapped_ids.at(computation.proto().entry_computation_id()));
  for (auto& imported_computation : imported_computations) {
    for (auto& instruction : *imported_computation.mutable_instructions()) {
      for (auto& operand_id : *instruction.mutable_operand_ids()) {
        operand_id = remapped_ids.at(operand_id);
      }
      for (auto& control_predecessor_id :
           *instruction.mutable_control_predecessor_ids()) {
        control_predecessor_id = remapped_ids.at(control_predecessor_id);
      }
      for (auto& called_computation_id :
           *instruction.mutable_called_computation_ids()) {
        called_computation_id = remapped_ids.at(called_computation_id);
      }
    }

    int64_t computation_id = imported_computation.id();
    for (int64_t i = 0; i < imported_computation.instructions_size(); ++i) {
      ImportedInstruction imported_instruction;
      imported_instruction.computation_id = computation_id;
      imported_instruction.instruction_index = i;
      handle_to_imported_index_.insert(
          {imported_computation.instructions(i).id(), imported_instruction});
    }
    embedded_.insert({computation_id, std::move(imported_computation)});
  }
}

StatusOr<const HloInstructionProto*> XlaBuilder::LookUpInstruction(
    const XlaOp op) const {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_188(mht_188_v, 4684, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::LookUpInstruction");

  TF_RETURN_IF_ERROR(first_error_);
  return LookUpInstructionInternal<const HloInstructionProto*>(op);
}

StatusOr<const HloInstructionProto*> XlaBuilder::LookUpInstructionByHandle(
    int64_t handle) const {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_189(mht_189_v, 4693, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::LookUpInstructionByHandle");

  return LookUpInstructionByHandleInternal<const HloInstructionProto*>(handle);
}

StatusOr<HloInstructionProto*> XlaBuilder::LookUpMutableInstruction(
    const XlaOp op) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_190(mht_190_v, 4701, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::LookUpMutableInstruction");

  TF_RETURN_IF_ERROR(first_error_);
  return LookUpInstructionInternal<HloInstructionProto*>(op);
}

StatusOr<HloInstructionProto*> XlaBuilder::LookUpMutableInstructionByHandle(
    int64_t handle) {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_191(mht_191_v, 4710, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "XlaBuilder::LookUpMutableInstructionByHandle");

  return LookUpInstructionByHandleInternal<HloInstructionProto*>(handle);
}

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
XlaOp Parameter(XlaBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name) {
   std::vector<std::string> mht_192_v;
   mht_192_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_192(mht_192_v, 4721, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Parameter");

  std::vector<bool> empty_bools;
  return Parameter(builder, parameter_number, shape, name, empty_bools);
}

XlaOp Parameter(XlaBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name,
                const std::vector<bool>& replicated_at_leaf_buffers) {
   std::vector<std::string> mht_193_v;
   mht_193_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_193(mht_193_v, 4732, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Parameter");

  return builder->Parameter(parameter_number, shape, name,
                            replicated_at_leaf_buffers);
}

// Enqueues a constant with the value of the given literal onto the
// computation.
XlaOp ConstantLiteral(XlaBuilder* builder, const LiteralSlice& literal) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_194(mht_194_v, 4742, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConstantLiteral");

  return builder->ConstantLiteral(literal);
}

XlaOp Broadcast(const XlaOp operand,
                absl::Span<const int64_t> broadcast_sizes) {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_195(mht_195_v, 4750, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Broadcast");

  return operand.builder()->Broadcast(operand, broadcast_sizes);
}

XlaOp BroadcastInDim(const XlaOp operand,
                     const absl::Span<const int64_t> out_dim_size,
                     const absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_196(mht_196_v, 4759, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "BroadcastInDim");

  return operand.builder()->BroadcastInDim(operand, out_dim_size,
                                           broadcast_dimensions);
}

XlaOp Copy(const XlaOp operand) {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_197(mht_197_v, 4767, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Copy");

  return operand.builder()->UnaryOp(HloOpcode::kCopy, operand);
}

XlaOp Pad(const XlaOp operand, const XlaOp padding_value,
          const PaddingConfig& padding_config) {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_198(mht_198_v, 4775, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Pad");

  return operand.builder()->Pad(operand, padding_value, padding_config);
}

XlaOp PadInDim(XlaOp operand, XlaOp padding_value, int64_t dimno,
               int64_t pad_lo, int64_t pad_hi) {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_199(mht_199_v, 4783, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "PadInDim");

  return operand.builder()->PadInDim(operand, padding_value, dimno, pad_lo,
                                     pad_hi);
}

XlaOp Reshape(const XlaOp operand, absl::Span<const int64_t> dimensions,
              absl::Span<const int64_t> new_sizes) {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_200(mht_200_v, 4792, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Reshape");

  return operand.builder()->Reshape(operand, dimensions, new_sizes);
}

XlaOp Reshape(const XlaOp operand, absl::Span<const int64_t> new_sizes) {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_201(mht_201_v, 4799, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Reshape");

  return operand.builder()->Reshape(operand, new_sizes);
}

XlaOp Reshape(const Shape& shape, XlaOp operand) {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_202(mht_202_v, 4806, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Reshape");

  return operand.builder()->Reshape(shape, operand);
}

XlaOp DynamicReshape(XlaOp operand, absl::Span<const XlaOp> dim_sizes,
                     absl::Span<const int64_t> new_size_bounds,
                     const std::vector<bool>& dims_are_dynamic) {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_203(mht_203_v, 4815, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicReshape");

  return operand.builder()->DynamicReshape(operand, dim_sizes, new_size_bounds,
                                           dims_are_dynamic);
}

XlaOp ReshapeWithInferredDimension(XlaOp operand,
                                   absl::Span<const int64_t> new_sizes,
                                   int64_t inferred_dimension) {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_204(mht_204_v, 4825, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReshapeWithInferredDimension");

  return operand.builder()->Reshape(operand, new_sizes, inferred_dimension);
}

XlaOp Collapse(const XlaOp operand, absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_205_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_205(mht_205_v, 4832, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Collapse");

  return operand.builder()->Collapse(operand, dimensions);
}

XlaOp Slice(const XlaOp operand, absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> limit_indices,
            absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_206(mht_206_v, 4841, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Slice");

  return operand.builder()->Slice(operand, start_indices, limit_indices,
                                  strides);
}

XlaOp SliceInDim(const XlaOp operand, int64_t start_index, int64_t limit_index,
                 int64_t stride, int64_t dimno) {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_207(mht_207_v, 4850, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SliceInDim");

  return operand.builder()->SliceInDim(operand, start_index, limit_index,
                                       stride, dimno);
}

XlaOp DynamicSlice(const XlaOp operand, absl::Span<const XlaOp> start_indices,
                   absl::Span<const int64_t> slice_sizes) {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_208(mht_208_v, 4859, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicSlice");

  return operand.builder()->DynamicSlice(operand, start_indices, slice_sizes);
}

XlaOp DynamicUpdateSlice(const XlaOp operand, const XlaOp update,
                         absl::Span<const XlaOp> start_indices) {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_209(mht_209_v, 4867, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicUpdateSlice");

  return operand.builder()->DynamicUpdateSlice(operand, update, start_indices);
}

XlaOp ConcatInDim(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                  int64_t dimension) {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_210(mht_210_v, 4875, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConcatInDim");

  return builder->ConcatInDim(operands, dimension);
}

void Trace(const std::string& tag, const XlaOp operand) {
   std::vector<std::string> mht_211_v;
   mht_211_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_211(mht_211_v, 4883, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Trace");

  return operand.builder()->Trace(tag, operand);
}

XlaOp Select(const XlaOp pred, const XlaOp on_true, const XlaOp on_false) {
   std::vector<std::string> mht_212_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_212(mht_212_v, 4890, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Select");

  return pred.builder()->Select(pred, on_true, on_false);
}

XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements) {
   std::vector<std::string> mht_213_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_213(mht_213_v, 4897, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Tuple");

  return builder->Tuple(elements);
}

XlaOp GetTupleElement(const XlaOp tuple_data, int64_t index) {
   std::vector<std::string> mht_214_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_214(mht_214_v, 4904, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GetTupleElement");

  return tuple_data.builder()->GetTupleElement(tuple_data, index);
}

XlaOp Eq(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_215_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_215(mht_215_v, 4912, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Eq");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kEq);
}

static XlaOp CompareTotalOrder(const XlaOp lhs, const XlaOp rhs,
                               absl::Span<const int64_t> broadcast_dimensions,
                               ComparisonDirection comparison_direction) {
   std::vector<std::string> mht_216_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_216(mht_216_v, 4921, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CompareTotalOrder");

  auto b = lhs.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto operand_shape, b->GetShape(lhs));
    auto operand_element_type = operand_shape.element_type();
    auto compare_type =
        primitive_util::IsFloatingPointType(operand_element_type)
            ? Comparison::Type::kFloatTotalOrder
            : Comparison::DefaultComparisonType(operand_element_type);
    return Compare(lhs, rhs, broadcast_dimensions, comparison_direction,
                   compare_type);
  });
}

XlaOp EqTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_217_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_217(mht_217_v, 4939, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "EqTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kEq);
}

XlaOp Ne(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_218_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_218(mht_218_v, 4948, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Ne");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kNe);
}

XlaOp NeTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_219_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_219(mht_219_v, 4956, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "NeTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kNe);
}

XlaOp Ge(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_220_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_220(mht_220_v, 4965, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Ge");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kGe);
}

XlaOp GeTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_221_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_221(mht_221_v, 4973, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GeTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kGe);
}

XlaOp Gt(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_222_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_222(mht_222_v, 4982, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Gt");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kGt);
}

XlaOp GtTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_223_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_223(mht_223_v, 4990, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GtTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kGt);
}

XlaOp Le(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_224_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_224(mht_224_v, 4999, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Le");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kLe);
}

XlaOp LeTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_225_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_225(mht_225_v, 5007, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "LeTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kLe);
}

XlaOp Lt(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_226_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_226(mht_226_v, 5016, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Lt");

  return Compare(lhs, rhs, broadcast_dimensions, ComparisonDirection::kLt);
}

XlaOp LtTotalOrder(const XlaOp lhs, const XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_227_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_227(mht_227_v, 5024, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "LtTotalOrder");

  return CompareTotalOrder(lhs, rhs, broadcast_dimensions,
                           ComparisonDirection::kLt);
}

XlaOp Compare(const XlaOp lhs, const XlaOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction) {
   std::vector<std::string> mht_228_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_228(mht_228_v, 5034, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Compare");

  return lhs.builder()->BinaryOp(HloOpcode::kCompare, lhs, rhs,
                                 broadcast_dimensions, direction);
}

XlaOp Compare(const XlaOp lhs, const XlaOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction, Comparison::Type compare_type) {
   std::vector<std::string> mht_229_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_229(mht_229_v, 5044, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Compare");

  return lhs.builder()->BinaryOp(HloOpcode::kCompare, lhs, rhs,
                                 broadcast_dimensions, direction, compare_type);
}

XlaOp Compare(const XlaOp lhs, const XlaOp rhs, ComparisonDirection direction) {
   std::vector<std::string> mht_230_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_230(mht_230_v, 5052, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Compare");

  return Compare(lhs, rhs, {}, direction);
}

XlaOp Dot(const XlaOp lhs, const XlaOp rhs,
          const PrecisionConfig* precision_config,
          absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_231_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_231(mht_231_v, 5061, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Dot");

  return lhs.builder()->Dot(lhs, rhs, precision_config, preferred_element_type);
}

XlaOp DotGeneral(const XlaOp lhs, const XlaOp rhs,
                 const DotDimensionNumbers& dimension_numbers,
                 const PrecisionConfig* precision_config,
                 absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_232_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_232(mht_232_v, 5071, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DotGeneral");

  return lhs.builder()->DotGeneral(lhs, rhs, dimension_numbers,
                                   precision_config, preferred_element_type);
}

XlaOp Conv(const XlaOp lhs, const XlaOp rhs,
           absl::Span<const int64_t> window_strides, Padding padding,
           int64_t feature_group_count, int64_t batch_group_count,
           const PrecisionConfig* precision_config,
           absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_233_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_233(mht_233_v, 5083, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Conv");

  return lhs.builder()->Conv(lhs, rhs, window_strides, padding,
                             feature_group_count, batch_group_count,
                             precision_config, preferred_element_type);
}

XlaOp ConvWithGeneralPadding(
    const XlaOp lhs, const XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_234_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_234(mht_234_v, 5097, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConvWithGeneralPadding");

  return lhs.builder()->ConvWithGeneralPadding(
      lhs, rhs, window_strides, padding, feature_group_count, batch_group_count,
      precision_config, preferred_element_type);
}

XlaOp ConvWithGeneralDimensions(
    const XlaOp lhs, const XlaOp rhs, absl::Span<const int64_t> window_strides,
    Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_235_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_235(mht_235_v, 5111, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConvWithGeneralDimensions");

  return lhs.builder()->ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding, dimension_numbers, feature_group_count,
      batch_group_count, precision_config, preferred_element_type);
}

XlaOp ConvGeneral(const XlaOp lhs, const XlaOp rhs,
                  absl::Span<const int64_t> window_strides,
                  absl::Span<const std::pair<int64_t, int64_t>> padding,
                  const ConvolutionDimensionNumbers& dimension_numbers,
                  int64_t feature_group_count, int64_t batch_group_count,
                  const PrecisionConfig* precision_config,
                  absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_236_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_236(mht_236_v, 5126, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConvGeneral");

  return lhs.builder()->ConvGeneral(
      lhs, rhs, window_strides, padding, dimension_numbers, feature_group_count,
      batch_group_count, precision_config, preferred_element_type);
}

XlaOp ConvGeneralDilated(const XlaOp lhs, const XlaOp rhs,
                         absl::Span<const int64_t> window_strides,
                         absl::Span<const std::pair<int64_t, int64_t>> padding,
                         absl::Span<const int64_t> lhs_dilation,
                         absl::Span<const int64_t> rhs_dilation,
                         const ConvolutionDimensionNumbers& dimension_numbers,
                         int64_t feature_group_count, int64_t batch_group_count,
                         const PrecisionConfig* precision_config,
                         absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_237_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_237(mht_237_v, 5143, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConvGeneralDilated");

  return lhs.builder()->ConvGeneralDilated(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, batch_group_count,
      precision_config, preferred_element_type);
}

XlaOp DynamicConvInputGrad(
    XlaOp input_sizes, const XlaOp lhs, const XlaOp rhs,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_238_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_238(mht_238_v, 5162, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicConvInputGrad");

  return lhs.builder()->DynamicConvInputGrad(
      input_sizes, lhs, rhs, window_strides, padding, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count,
      precision_config, padding_type, preferred_element_type);
}

XlaOp DynamicConvKernelGrad(
    XlaOp activations, XlaOp gradients,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_239_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_239(mht_239_v, 5181, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicConvKernelGrad");

  return activations.builder()->DynamicConvKernelGrad(
      activations, gradients, window_strides, padding, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count,
      precision_config, padding_type, preferred_element_type);
}

XlaOp DynamicConvForward(const XlaOp lhs, const XlaOp rhs,
                         absl::Span<const int64_t> window_strides,
                         absl::Span<const std::pair<int64_t, int64_t>> padding,
                         absl::Span<const int64_t> lhs_dilation,
                         absl::Span<const int64_t> rhs_dilation,
                         const ConvolutionDimensionNumbers& dimension_numbers,
                         int64_t feature_group_count, int64_t batch_group_count,
                         const PrecisionConfig* precision_config,
                         PaddingType padding_type,
                         absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_240_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_240(mht_240_v, 5200, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "DynamicConvForward");

  return lhs.builder()->DynamicConvForward(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, batch_group_count,
      precision_config, padding_type, preferred_element_type);
}

XlaOp Fft(const XlaOp operand, FftType fft_type,
          absl::Span<const int64_t> fft_length) {
   std::vector<std::string> mht_241_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_241(mht_241_v, 5211, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Fft");

  return operand.builder()->Fft(operand, fft_type, fft_length);
}

XlaOp TriangularSolve(XlaOp a, XlaOp b, bool left_side, bool lower,
                      bool unit_diagonal,
                      TriangularSolveOptions::Transpose transpose_a) {
   std::vector<std::string> mht_242_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_242(mht_242_v, 5220, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "TriangularSolve");

  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* a_shape, builder->GetShapePtr(a));
    TF_ASSIGN_OR_RETURN(const Shape* b_shape, builder->GetShapePtr(b));
    xla::TriangularSolveOptions options;
    options.set_left_side(left_side);
    options.set_lower(lower);
    options.set_unit_diagonal(unit_diagonal);
    options.set_transpose_a(transpose_a);
    TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferTriangularSolveShape(
                                         *a_shape, *b_shape, options));
    return builder->TriangularSolveInternal(shape, a, b, std::move(options));
  });
}

XlaOp Cholesky(XlaOp a, bool lower) {
   std::vector<std::string> mht_243_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_243(mht_243_v, 5239, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Cholesky");

  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* a_shape, builder->GetShapePtr(a));
    TF_ASSIGN_OR_RETURN(Shape shape,
                        ShapeInference::InferCholeskyShape(*a_shape));
    return builder->CholeskyInternal(shape, a, lower);
  });
}

XlaOp Infeed(XlaBuilder* builder, const Shape& shape,
             const std::string& config) {
   std::vector<std::string> mht_244_v;
   mht_244_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_244(mht_244_v, 5254, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Infeed");

  return builder->Infeed(shape, config);
}

void Outfeed(const XlaOp operand, const Shape& shape_with_layout,
             const std::string& outfeed_config) {
   std::vector<std::string> mht_245_v;
   mht_245_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_245(mht_245_v, 5263, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Outfeed");

  return operand.builder()->Outfeed(operand, shape_with_layout, outfeed_config);
}

XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
           absl::Span<const XlaOp> operands) {
   std::vector<std::string> mht_246_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_246(mht_246_v, 5271, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Call");

  return builder->Call(computation, operands);
}

XlaOp CustomCall(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape,
    const std::string& opaque, bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, CustomCallSchedule schedule,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_247_v;
   mht_247_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_247_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_247(mht_247_v, 5287, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CustomCall");

  return builder->CustomCall(call_target_name, operands, shape, opaque,
                             /*operand_shapes_with_layout=*/absl::nullopt,
                             has_side_effect, output_operand_aliasing, literal,
                             /*window=*/absl::nullopt, /*dnums=*/absl::nullopt,
                             schedule, api_version);
}

XlaOp CustomCallWithComputation(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const XlaComputation& computation,
    const Shape& shape, const std::string& opaque, bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, CustomCallSchedule schedule,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_248_v;
   mht_248_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_248_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_248(mht_248_v, 5307, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CustomCallWithComputation");

  return builder->CustomCall(
      call_target_name, operands, computation, shape, opaque,
      /*operand_shapes_with_layout=*/absl::nullopt, has_side_effect,
      output_operand_aliasing, literal, schedule, api_version);
}

XlaOp CustomCallWithLayout(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape,
    absl::Span<const Shape> operand_shapes_with_layout,
    const std::string& opaque, bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, CustomCallSchedule schedule,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_249_v;
   mht_249_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_249_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_249(mht_249_v, 5327, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CustomCallWithLayout");

  return builder->CustomCall(
      call_target_name, operands, shape, opaque, operand_shapes_with_layout,
      has_side_effect, output_operand_aliasing, literal,
      /*window=*/absl::nullopt, /*dnums=*/absl::nullopt, schedule, api_version);
}

XlaOp CustomCallWithConvDnums(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape,
    absl::Span<const Shape> operand_shapes_with_layout,
    const std::string& opaque, bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, Window window, ConvolutionDimensionNumbers dnums,
    CustomCallSchedule schedule, CustomCallApiVersion api_version) {
   std::vector<std::string> mht_250_v;
   mht_250_v.push_back("call_target_name: \"" + call_target_name + "\"");
   mht_250_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_250(mht_250_v, 5347, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CustomCallWithConvDnums");

  absl::optional<absl::Span<const Shape>> maybe_operand_shapes;
  if (!operand_shapes_with_layout.empty()) {
    maybe_operand_shapes = operand_shapes_with_layout;
  }
  return builder->CustomCall(call_target_name, operands, shape, opaque,
                             maybe_operand_shapes, has_side_effect,
                             output_operand_aliasing, literal, window, dnums,
                             schedule, api_version);
}

XlaOp OptimizationBarrier(XlaOp operand) {
   std::vector<std::string> mht_251_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_251(mht_251_v, 5361, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "OptimizationBarrier");

  return operand.builder()->OptimizationBarrier(operand);
}

XlaOp Complex(const XlaOp lhs, const XlaOp rhs,
              absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_252_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_252(mht_252_v, 5369, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Complex");

  return lhs.builder()->BinaryOp(HloOpcode::kComplex, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Conj(const XlaOp operand) {
   std::vector<std::string> mht_253_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_253(mht_253_v, 5377, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Conj");

  return Complex(Real(operand), Neg(Imag(operand)));
}

XlaOp Add(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_254_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_254(mht_254_v, 5385, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Add");

  return lhs.builder()->BinaryOp(HloOpcode::kAdd, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Sub(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_255_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_255(mht_255_v, 5394, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Sub");

  return lhs.builder()->BinaryOp(HloOpcode::kSubtract, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Mul(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_256_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_256(mht_256_v, 5403, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Mul");

  return lhs.builder()->BinaryOp(HloOpcode::kMultiply, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Div(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_257_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_257(mht_257_v, 5412, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Div");

  return lhs.builder()->BinaryOp(HloOpcode::kDivide, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Rem(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_258_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_258(mht_258_v, 5421, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Rem");

  return lhs.builder()->BinaryOp(HloOpcode::kRemainder, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Max(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_259_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_259(mht_259_v, 5430, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Max");

  return lhs.builder()->BinaryOp(HloOpcode::kMaximum, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Min(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_260_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_260(mht_260_v, 5439, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Min");

  return lhs.builder()->BinaryOp(HloOpcode::kMinimum, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp And(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_261_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_261(mht_261_v, 5448, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "And");

  return lhs.builder()->BinaryOp(HloOpcode::kAnd, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Or(const XlaOp lhs, const XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_262_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_262(mht_262_v, 5457, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Or");

  return lhs.builder()->BinaryOp(HloOpcode::kOr, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Xor(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_263_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_263(mht_263_v, 5466, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Xor");

  return lhs.builder()->BinaryOp(HloOpcode::kXor, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Not(const XlaOp operand) {
   std::vector<std::string> mht_264_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_264(mht_264_v, 5474, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Not");

  return operand.builder()->UnaryOp(HloOpcode::kNot, operand);
}

XlaOp PopulationCount(const XlaOp operand) {
   std::vector<std::string> mht_265_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_265(mht_265_v, 5481, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "PopulationCount");

  return operand.builder()->UnaryOp(HloOpcode::kPopulationCount, operand);
}

XlaOp ShiftLeft(const XlaOp lhs, const XlaOp rhs,
                absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_266_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_266(mht_266_v, 5489, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ShiftLeft");

  return lhs.builder()->BinaryOp(HloOpcode::kShiftLeft, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp ShiftRightArithmetic(const XlaOp lhs, const XlaOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_267_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_267(mht_267_v, 5498, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ShiftRightArithmetic");

  return lhs.builder()->BinaryOp(HloOpcode::kShiftRightArithmetic, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp ShiftRightLogical(const XlaOp lhs, const XlaOp rhs,
                        absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_268_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_268(mht_268_v, 5507, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ShiftRightLogical");

  return lhs.builder()->BinaryOp(HloOpcode::kShiftRightLogical, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Reduce(const XlaOp operand, const XlaOp init_value,
             const XlaComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_269_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_269(mht_269_v, 5517, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Reduce");

  return operand.builder()->Reduce(operand, init_value, computation,
                                   dimensions_to_reduce);
}

// Reduces several arrays simultaneously among the provided dimensions, given
// "computation" as a reduction operator.
XlaOp Reduce(XlaBuilder* builder, absl::Span<const XlaOp> operands,
             absl::Span<const XlaOp> init_values,
             const XlaComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce) {
   std::vector<std::string> mht_270_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_270(mht_270_v, 5530, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Reduce");

  return builder->Reduce(operands, init_values, computation,
                         dimensions_to_reduce);
}

XlaOp ReduceAll(const XlaOp operand, const XlaOp init_value,
                const XlaComputation& computation) {
   std::vector<std::string> mht_271_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_271(mht_271_v, 5539, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceAll");

  return operand.builder()->ReduceAll(operand, init_value, computation);
}

XlaOp ReduceWindow(const XlaOp operand, const XlaOp init_value,
                   const XlaComputation& computation,
                   absl::Span<const int64_t> window_dimensions,
                   absl::Span<const int64_t> window_strides, Padding padding) {
   std::vector<std::string> mht_272_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_272(mht_272_v, 5549, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceWindow");

  return operand.builder()->ReduceWindow(operand, init_value, computation,
                                         window_dimensions, window_strides,
                                         padding);
}

XlaOp ReduceWindow(absl::Span<const XlaOp> operands,
                   absl::Span<const XlaOp> init_values,
                   const XlaComputation& computation,
                   absl::Span<const int64_t> window_dimensions,
                   absl::Span<const int64_t> window_strides, Padding padding) {
   std::vector<std::string> mht_273_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_273(mht_273_v, 5562, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceWindow");

  CHECK(!operands.empty());
  return operands[0].builder()->ReduceWindow(operands, init_values, computation,
                                             window_dimensions, window_strides,
                                             padding);
}

XlaOp ReduceWindowWithGeneralPadding(
    const XlaOp operand, const XlaOp init_value,
    const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
   std::vector<std::string> mht_274_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_274(mht_274_v, 5579, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceWindowWithGeneralPadding");

  return operand.builder()->ReduceWindowWithGeneralPadding(
      absl::MakeSpan(&operand, 1), absl::MakeSpan(&init_value, 1), computation,
      window_dimensions, window_strides, base_dilations, window_dilations,
      padding);
}

XlaOp ReduceWindowWithGeneralPadding(
    absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
    const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
   std::vector<std::string> mht_275_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_275(mht_275_v, 5596, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceWindowWithGeneralPadding");

  CHECK(!operands.empty());
  return operands[0].builder()->ReduceWindowWithGeneralPadding(
      operands, init_values, computation, window_dimensions, window_strides,
      base_dilations, window_dilations, padding);
}

XlaOp AllGather(const XlaOp operand, int64_t all_gather_dimension,
                int64_t shard_count,
                absl::Span<const ReplicaGroup> replica_groups,
                const absl::optional<ChannelHandle>& channel_id,
                const absl::optional<Layout>& layout,
                const absl::optional<bool> use_global_device_ids) {
   std::vector<std::string> mht_276_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_276(mht_276_v, 5611, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "AllGather");

  return operand.builder()->AllGather(operand, all_gather_dimension,
                                      shard_count, replica_groups, channel_id,
                                      layout, use_global_device_ids);
}

XlaOp CrossReplicaSum(const XlaOp operand,
                      absl::Span<const ReplicaGroup> replica_groups) {
   std::vector<std::string> mht_277_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_277(mht_277_v, 5621, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CrossReplicaSum");

  return operand.builder()->CrossReplicaSum(operand, replica_groups);
}

XlaOp AllReduce(const XlaOp operand, const XlaComputation& computation,
                absl::Span<const ReplicaGroup> replica_groups,
                const absl::optional<ChannelHandle>& channel_id,
                const absl::optional<Shape>& shape_with_layout) {
   std::vector<std::string> mht_278_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_278(mht_278_v, 5631, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "AllReduce");

  return operand.builder()->AllReduce(operand, computation, replica_groups,
                                      channel_id, shape_with_layout);
}

XlaOp ReduceScatter(const XlaOp operand, const XlaComputation& computation,
                    int64_t scatter_dimension, int64_t shard_count,
                    absl::Span<const ReplicaGroup> replica_groups,
                    const absl::optional<ChannelHandle>& channel_id,
                    const absl::optional<Layout>& layout,
                    const absl::optional<bool> use_global_device_ids) {
   std::vector<std::string> mht_279_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_279(mht_279_v, 5644, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReduceScatter");

  return operand.builder()->ReduceScatter(
      operand, computation, scatter_dimension, shard_count, replica_groups,
      channel_id, layout, use_global_device_ids);
}

XlaOp AllToAll(const XlaOp operand, int64_t split_dimension,
               int64_t concat_dimension, int64_t split_count,
               absl::Span<const ReplicaGroup> replica_groups,
               const absl::optional<Layout>& layout) {
   std::vector<std::string> mht_280_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_280(mht_280_v, 5656, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "AllToAll");

  return operand.builder()->AllToAll(operand, split_dimension, concat_dimension,
                                     split_count, replica_groups, layout);
}

XlaOp AllToAllTuple(const XlaOp operand, int64_t split_dimension,
                    int64_t concat_dimension, int64_t split_count,
                    absl::Span<const ReplicaGroup> replica_groups,
                    const absl::optional<Layout>& layout) {
   std::vector<std::string> mht_281_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_281(mht_281_v, 5667, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "AllToAllTuple");

  return operand.builder()->AllToAllTuple(operand, split_dimension,
                                          concat_dimension, split_count,
                                          replica_groups, layout);
}

XlaOp CollectivePermute(
    const XlaOp operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
   std::vector<std::string> mht_282_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_282(mht_282_v, 5678, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CollectivePermute");

  return operand.builder()->CollectivePermute(operand, source_target_pairs);
}

XlaOp ReplicaId(XlaBuilder* builder) {
   std::vector<std::string> mht_283_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_283(mht_283_v, 5685, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReplicaId");
 return builder->ReplicaId(); }

XlaOp SelectAndScatter(const XlaOp operand, const XlaComputation& select,
                       absl::Span<const int64_t> window_dimensions,
                       absl::Span<const int64_t> window_strides,
                       Padding padding, const XlaOp source,
                       const XlaOp init_value, const XlaComputation& scatter) {
   std::vector<std::string> mht_284_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_284(mht_284_v, 5694, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SelectAndScatter");

  return operand.builder()->SelectAndScatter(operand, select, window_dimensions,
                                             window_strides, padding, source,
                                             init_value, scatter);
}

XlaOp SelectAndScatterWithGeneralPadding(
    const XlaOp operand, const XlaComputation& select,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding, const XlaOp source,
    const XlaOp init_value, const XlaComputation& scatter) {
   std::vector<std::string> mht_285_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_285(mht_285_v, 5708, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SelectAndScatterWithGeneralPadding");

  return operand.builder()->SelectAndScatterWithGeneralPadding(
      operand, select, window_dimensions, window_strides, padding, source,
      init_value, scatter);
}

XlaOp Abs(const XlaOp operand) {
   std::vector<std::string> mht_286_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_286(mht_286_v, 5717, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Abs");

  return operand.builder()->UnaryOp(HloOpcode::kAbs, operand);
}

XlaOp Atan2(const XlaOp lhs, const XlaOp rhs,
            absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_287_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_287(mht_287_v, 5725, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Atan2");

  return lhs.builder()->BinaryOp(HloOpcode::kAtan2, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp Exp(const XlaOp operand) {
   std::vector<std::string> mht_288_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_288(mht_288_v, 5733, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Exp");

  return operand.builder()->UnaryOp(HloOpcode::kExp, operand);
}
XlaOp Expm1(const XlaOp operand) {
   std::vector<std::string> mht_289_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_289(mht_289_v, 5739, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Expm1");

  return operand.builder()->UnaryOp(HloOpcode::kExpm1, operand);
}
XlaOp Floor(const XlaOp operand) {
   std::vector<std::string> mht_290_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_290(mht_290_v, 5745, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Floor");

  return operand.builder()->UnaryOp(HloOpcode::kFloor, operand);
}
XlaOp Ceil(const XlaOp operand) {
   std::vector<std::string> mht_291_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_291(mht_291_v, 5751, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Ceil");

  return operand.builder()->UnaryOp(HloOpcode::kCeil, operand);
}
XlaOp Round(const XlaOp operand) {
   std::vector<std::string> mht_292_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_292(mht_292_v, 5757, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Round");

  return operand.builder()->UnaryOp(HloOpcode::kRoundNearestAfz, operand);
}
XlaOp Log(const XlaOp operand) {
   std::vector<std::string> mht_293_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_293(mht_293_v, 5763, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Log");

  return operand.builder()->UnaryOp(HloOpcode::kLog, operand);
}
XlaOp Log1p(const XlaOp operand) {
   std::vector<std::string> mht_294_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_294(mht_294_v, 5769, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Log1p");

  return operand.builder()->UnaryOp(HloOpcode::kLog1p, operand);
}
XlaOp Logistic(const XlaOp operand) {
   std::vector<std::string> mht_295_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_295(mht_295_v, 5775, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Logistic");

  return operand.builder()->UnaryOp(HloOpcode::kLogistic, operand);
}
XlaOp Sign(const XlaOp operand) {
   std::vector<std::string> mht_296_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_296(mht_296_v, 5781, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Sign");

  return operand.builder()->UnaryOp(HloOpcode::kSign, operand);
}
XlaOp Clz(const XlaOp operand) {
   std::vector<std::string> mht_297_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_297(mht_297_v, 5787, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Clz");

  return operand.builder()->UnaryOp(HloOpcode::kClz, operand);
}
XlaOp Cos(const XlaOp operand) {
   std::vector<std::string> mht_298_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_298(mht_298_v, 5793, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Cos");

  return operand.builder()->UnaryOp(HloOpcode::kCos, operand);
}
XlaOp Sin(const XlaOp operand) {
   std::vector<std::string> mht_299_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_299(mht_299_v, 5799, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Sin");

  return operand.builder()->UnaryOp(HloOpcode::kSin, operand);
}
XlaOp Tanh(const XlaOp operand) {
   std::vector<std::string> mht_300_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_300(mht_300_v, 5805, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Tanh");

  return operand.builder()->UnaryOp(HloOpcode::kTanh, operand);
}
XlaOp Real(const XlaOp operand) {
   std::vector<std::string> mht_301_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_301(mht_301_v, 5811, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Real");

  return operand.builder()->UnaryOp(HloOpcode::kReal, operand);
}
XlaOp Imag(const XlaOp operand) {
   std::vector<std::string> mht_302_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_302(mht_302_v, 5817, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Imag");

  return operand.builder()->UnaryOp(HloOpcode::kImag, operand);
}
XlaOp Sqrt(const XlaOp operand) {
   std::vector<std::string> mht_303_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_303(mht_303_v, 5823, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Sqrt");

  return operand.builder()->UnaryOp(HloOpcode::kSqrt, operand);
}
XlaOp Cbrt(const XlaOp operand) {
   std::vector<std::string> mht_304_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_304(mht_304_v, 5829, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Cbrt");

  return operand.builder()->UnaryOp(HloOpcode::kCbrt, operand);
}
XlaOp Rsqrt(const XlaOp operand) {
   std::vector<std::string> mht_305_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_305(mht_305_v, 5835, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Rsqrt");

  return operand.builder()->UnaryOp(HloOpcode::kRsqrt, operand);
}

XlaOp Pow(const XlaOp lhs, const XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_306_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_306(mht_306_v, 5843, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Pow");

  return lhs.builder()->BinaryOp(HloOpcode::kPower, lhs, rhs,
                                 broadcast_dimensions);
}

XlaOp IsFinite(const XlaOp operand) {
   std::vector<std::string> mht_307_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_307(mht_307_v, 5851, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "IsFinite");

  return operand.builder()->UnaryOp(HloOpcode::kIsFinite, operand);
}

XlaOp ConvertElementType(const XlaOp operand, PrimitiveType new_element_type) {
   std::vector<std::string> mht_308_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_308(mht_308_v, 5858, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ConvertElementType");

  return operand.builder()->ConvertElementType(operand, new_element_type);
}

XlaOp BitcastConvertType(const XlaOp operand, PrimitiveType new_element_type) {
   std::vector<std::string> mht_309_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_309(mht_309_v, 5865, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "BitcastConvertType");

  return operand.builder()->BitcastConvertType(operand, new_element_type);
}

XlaOp Neg(const XlaOp operand) {
   std::vector<std::string> mht_310_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_310(mht_310_v, 5872, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Neg");

  return operand.builder()->UnaryOp(HloOpcode::kNegate, operand);
}

XlaOp Transpose(const XlaOp operand, absl::Span<const int64_t> permutation) {
   std::vector<std::string> mht_311_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_311(mht_311_v, 5879, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Transpose");

  return operand.builder()->Transpose(operand, permutation);
}

XlaOp Rev(const XlaOp operand, absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_312_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_312(mht_312_v, 5886, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Rev");

  return operand.builder()->Rev(operand, dimensions);
}

XlaOp Sort(absl::Span<const XlaOp> operands, const XlaComputation& comparator,
           int64_t dimension, bool is_stable) {
   std::vector<std::string> mht_313_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_313(mht_313_v, 5894, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Sort");

  return operands[0].builder()->Sort(operands, comparator, dimension,
                                     is_stable);
}

XlaOp Clamp(const XlaOp min, const XlaOp operand, const XlaOp max) {
   std::vector<std::string> mht_314_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_314(mht_314_v, 5902, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Clamp");

  return min.builder()->Clamp(min, operand, max);
}

XlaOp Map(XlaBuilder* builder, absl::Span<const XlaOp> operands,
          const XlaComputation& computation,
          absl::Span<const int64_t> dimensions,
          absl::Span<const XlaOp> static_operands) {
   std::vector<std::string> mht_315_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_315(mht_315_v, 5912, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Map");

  return builder->Map(operands, computation, dimensions, static_operands);
}

XlaOp RngNormal(const XlaOp mu, const XlaOp sigma, const Shape& shape) {
   std::vector<std::string> mht_316_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_316(mht_316_v, 5919, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RngNormal");

  return mu.builder()->RngNormal(mu, sigma, shape);
}

XlaOp RngUniform(const XlaOp a, const XlaOp b, const Shape& shape) {
   std::vector<std::string> mht_317_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_317(mht_317_v, 5926, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RngUniform");

  return a.builder()->RngUniform(a, b, shape);
}

XlaOp RngBitGenerator(RandomAlgorithm algorithm, const XlaOp initial_state,
                      const Shape& shape) {
   std::vector<std::string> mht_318_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_318(mht_318_v, 5934, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RngBitGenerator");

  return initial_state.builder()->RngBitGenerator(algorithm, initial_state,
                                                  shape);
}

XlaOp While(const XlaComputation& condition, const XlaComputation& body,
            const XlaOp init) {
   std::vector<std::string> mht_319_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_319(mht_319_v, 5943, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "While");

  return init.builder()->While(condition, body, init);
}

XlaOp Conditional(const XlaOp predicate, const XlaOp true_operand,
                  const XlaComputation& true_computation,
                  const XlaOp false_operand,
                  const XlaComputation& false_computation) {
   std::vector<std::string> mht_320_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_320(mht_320_v, 5953, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Conditional");

  return predicate.builder()->Conditional(predicate, true_operand,
                                          true_computation, false_operand,
                                          false_computation);
}

XlaOp Conditional(const XlaOp branch_index,
                  absl::Span<const XlaComputation* const> branch_computations,
                  absl::Span<const XlaOp> branch_operands) {
   std::vector<std::string> mht_321_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_321(mht_321_v, 5964, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Conditional");

  return branch_index.builder()->Conditional(branch_index, branch_computations,
                                             branch_operands);
}

XlaOp ReducePrecision(const XlaOp operand, const int exponent_bits,
                      const int mantissa_bits) {
   std::vector<std::string> mht_322_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_322(mht_322_v, 5973, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "ReducePrecision");

  return operand.builder()->ReducePrecision(operand, exponent_bits,
                                            mantissa_bits);
}

XlaOp Gather(const XlaOp input, const XlaOp start_indices,
             const GatherDimensionNumbers& dimension_numbers,
             absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
   std::vector<std::string> mht_323_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_323(mht_323_v, 5983, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Gather");

  return input.builder()->Gather(input, start_indices, dimension_numbers,
                                 slice_sizes, indices_are_sorted);
}

XlaOp Scatter(const XlaOp input, const XlaOp scatter_indices,
              const XlaOp updates, const XlaComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted, bool unique_indices) {
   std::vector<std::string> mht_324_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_324(mht_324_v, 5994, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Scatter");

  return input.builder()->Scatter(input, scatter_indices, updates,
                                  update_computation, dimension_numbers,
                                  indices_are_sorted, unique_indices);
}

void Send(const XlaOp operand, const ChannelHandle& handle) {
   std::vector<std::string> mht_325_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_325(mht_325_v, 6003, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Send");

  return operand.builder()->Send(operand, handle);
}

XlaOp Recv(XlaBuilder* builder, const Shape& shape,
           const ChannelHandle& handle) {
   std::vector<std::string> mht_326_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_326(mht_326_v, 6011, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Recv");

  return builder->Recv(shape, handle);
}

XlaOp SendWithToken(const XlaOp operand, const XlaOp token,
                    const ChannelHandle& handle) {
   std::vector<std::string> mht_327_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_327(mht_327_v, 6019, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SendWithToken");

  return operand.builder()->SendWithToken(operand, token, handle);
}

XlaOp RecvWithToken(const XlaOp token, const Shape& shape,
                    const ChannelHandle& handle) {
   std::vector<std::string> mht_328_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_328(mht_328_v, 6027, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RecvWithToken");

  return token.builder()->RecvWithToken(token, shape, handle);
}

XlaOp SendToHost(const XlaOp operand, const XlaOp token,
                 const Shape& shape_with_layout, const ChannelHandle& handle) {
   std::vector<std::string> mht_329_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_329(mht_329_v, 6035, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SendToHost");

  return operand.builder()->SendToHost(operand, token, shape_with_layout,
                                       handle);
}

XlaOp RecvFromHost(const XlaOp token, const Shape& shape,
                   const ChannelHandle& handle) {
   std::vector<std::string> mht_330_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_330(mht_330_v, 6044, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RecvFromHost");

  return token.builder()->RecvFromHost(token, shape, handle);
}

XlaOp InfeedWithToken(const XlaOp token, const Shape& shape,
                      const std::string& config) {
   std::vector<std::string> mht_331_v;
   mht_331_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_331(mht_331_v, 6053, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "InfeedWithToken");

  return token.builder()->InfeedWithToken(token, shape, config);
}

XlaOp OutfeedWithToken(const XlaOp operand, const XlaOp token,
                       const Shape& shape_with_layout,
                       const std::string& outfeed_config) {
   std::vector<std::string> mht_332_v;
   mht_332_v.push_back("outfeed_config: \"" + outfeed_config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_332(mht_332_v, 6063, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "OutfeedWithToken");

  return operand.builder()->OutfeedWithToken(operand, token, shape_with_layout,
                                             outfeed_config);
}

XlaOp CreateToken(XlaBuilder* builder) {
   std::vector<std::string> mht_333_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_333(mht_333_v, 6071, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "CreateToken");
 return builder->CreateToken(); }

XlaOp AfterAll(XlaBuilder* builder, absl::Span<const XlaOp> tokens) {
   std::vector<std::string> mht_334_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_334(mht_334_v, 6076, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "AfterAll");

  return builder->AfterAll(tokens);
}

XlaOp BatchNormTraining(const XlaOp operand, const XlaOp scale,
                        const XlaOp offset, float epsilon,
                        int64_t feature_index) {
   std::vector<std::string> mht_335_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_335(mht_335_v, 6085, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "BatchNormTraining");

  return operand.builder()->BatchNormTraining(operand, scale, offset, epsilon,
                                              feature_index);
}

XlaOp BatchNormInference(const XlaOp operand, const XlaOp scale,
                         const XlaOp offset, const XlaOp mean,
                         const XlaOp variance, float epsilon,
                         int64_t feature_index) {
   std::vector<std::string> mht_336_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_336(mht_336_v, 6096, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "BatchNormInference");

  return operand.builder()->BatchNormInference(
      operand, scale, offset, mean, variance, epsilon, feature_index);
}

XlaOp BatchNormGrad(const XlaOp operand, const XlaOp scale,
                    const XlaOp batch_mean, const XlaOp batch_var,
                    const XlaOp grad_output, float epsilon,
                    int64_t feature_index) {
   std::vector<std::string> mht_337_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_337(mht_337_v, 6107, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "BatchNormGrad");

  return operand.builder()->BatchNormGrad(operand, scale, batch_mean, batch_var,
                                          grad_output, epsilon, feature_index);
}

XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64_t size) {
   std::vector<std::string> mht_338_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_338(mht_338_v, 6115, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Iota");

  return builder->Iota(type, size);
}

XlaOp Iota(XlaBuilder* builder, const Shape& shape, int64_t iota_dimension) {
   std::vector<std::string> mht_339_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_339(mht_339_v, 6122, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "Iota");

  return builder->Iota(shape, iota_dimension);
}

XlaOp GetDimensionSize(const XlaOp operand, int64_t dimension) {
   std::vector<std::string> mht_340_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_340(mht_340_v, 6129, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "GetDimensionSize");

  return operand.builder()->GetDimensionSize(operand, dimension);
}

XlaOp SetDimensionSize(const XlaOp operand, const XlaOp val,
                       int64_t dimension) {
   std::vector<std::string> mht_341_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_341(mht_341_v, 6137, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "SetDimensionSize");

  return operand.builder()->SetDimensionSize(operand, val, dimension);
}

XlaOp RemoveDynamicDimension(const XlaOp operand, int64_t dimension) {
   std::vector<std::string> mht_342_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSxla_builderDTcc mht_342(mht_342_v, 6144, "", "./tensorflow/compiler/xla/client/xla_builder.cc", "RemoveDynamicDimension");

  return operand.builder()->RemoveDynamicDimension(operand, dimension);
}

}  // namespace xla
