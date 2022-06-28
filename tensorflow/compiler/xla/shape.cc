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
class MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc() {
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

#include "tensorflow/compiler/xla/shape.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

Shape::Shape(const ShapeProto& shape_proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/shape.cc", "Shape::Shape");

  set_element_type(shape_proto.element_type());
  dimensions_.reserve(shape_proto.dimensions_size());
  for (const int64_t dimension : shape_proto.dimensions()) {
    add_dimensions(dimension);
  }
  // A malformed proto may have different is_dynamic_dimension_size and
  // dimensions_size. Since C++ is evil, and we have no good way of bailing out
  // in a constructor, conservatively trim the is_dynamic_dimension size.
  // TODO(b/120111794): Make this a hard error when we have a factory method
  // instead of a constructor.
  if (shape_proto.dimensions_size() !=
      shape_proto.is_dynamic_dimension_size()) {
    if (shape_proto.is_dynamic_dimension_size() != 0) {
      LOG(ERROR) << "Malformed shape proto: number of is_dynamic_dimension "
                    "fields does not match number of dimension fields";
    } else {
      LOG(WARNING) << "Malformed shape proto: is_dynamic_dimension is empty";
    }
  }
  int64_t num_dynamic_dimension_fields = std::min(
      shape_proto.dimensions_size(), shape_proto.is_dynamic_dimension_size());
  for (int i = 0; i < num_dynamic_dimension_fields; i++) {
    dynamic_dimensions_[i] = shape_proto.is_dynamic_dimension(i);
  }
  tuple_shapes_.reserve(shape_proto.tuple_shapes_size());
  for (const ShapeProto& element_shape : shape_proto.tuple_shapes()) {
    tuple_shapes_.emplace_back(element_shape);
  }
  if (shape_proto.has_layout()) {
    *mutable_layout() = Layout::CreateFromProto(shape_proto.layout());
  }
}

ShapeProto Shape::ToProto() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/shape.cc", "Shape::ToProto");

  ShapeProto proto;
  proto.set_element_type(element_type_);
  proto.mutable_dimensions()->Reserve(dimensions_size());
  for (const int64_t dimension : dimensions()) {
    proto.add_dimensions(dimension);
  }
  for (const bool dynamic : dynamic_dimensions_) {
    proto.add_is_dynamic_dimension(dynamic);
  }
  proto.mutable_tuple_shapes()->Reserve(tuple_shapes_size());
  for (const Shape& shape : tuple_shapes()) {
    *proto.add_tuple_shapes() = shape.ToProto();
  }
  if (has_layout()) {
    *proto.mutable_layout() = layout().ToProto();
  }
  return proto;
}

std::string Shape::ToString(bool print_layout) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/shape.cc", "Shape::ToString");

  if (print_layout) {
    return ShapeUtil::HumanStringWithLayout(*this);
  } else {
    return ShapeUtil::HumanString(*this);
  }
}

bool Shape::IsInteger() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/xla/shape.cc", "Shape::IsInteger");

  switch (element_type()) {
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
      return true;
    case PrimitiveType::TUPLE:
      return absl::c_any_of(tuple_shapes_,
                            [](const Shape& s) { return s.IsInteger(); });
    default:
      return false;
  }
}

bool Shape::is_static() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_4(mht_4_v, 286, "", "./tensorflow/compiler/xla/shape.cc", "Shape::is_static");

  if (IsTuple()) {
    for (const Shape& subshape : tuple_shapes_) {
      if (!subshape.is_static()) {
        return false;
      }
    }
  }
  return !absl::c_any_of(dynamic_dimensions_, [](bool b) { return b; });
}

void Shape::DeleteDimension(int64_t dim_to_delete) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_5(mht_5_v, 300, "", "./tensorflow/compiler/xla/shape.cc", "Shape::DeleteDimension");

  CHECK(IsArray());
  CHECK_GE(dim_to_delete, 0);
  CHECK_LT(dim_to_delete, dimensions_.size());
  dimensions_.erase(dimensions_.begin() + dim_to_delete);
  dynamic_dimensions_.erase(dynamic_dimensions_.begin() + dim_to_delete);
  if (LayoutUtil::HasLayout(*this)) {
    layout_.set_format(DENSE);
    for (int64_t i = 0; i < layout_.minor_to_major().size();) {
      if (layout_.minor_to_major(i) == dim_to_delete) {
        layout_.mutable_minor_to_major()->erase(
            layout_.mutable_minor_to_major()->begin() + i);
        continue;
      }
      if (layout_.minor_to_major(i) > dim_to_delete) {
        (*layout_.mutable_minor_to_major())[i] -= 1;
      }
      ++i;
    }
  }
}

bool Shape::Equal::operator()(const Shape& lhs, const Shape& rhs) {
  if (lhs.IsTuple()) {
    return rhs.IsTuple() &&
           absl::c_equal(
               lhs.tuple_shapes(), rhs.tuple_shapes(),
               [=](const Shape& l, const Shape& r) { return (*this)(l, r); });
  } else if (!lhs.IsArray()) {
    // Non-tuple, non-array tupes such as opaque and token types are trivially
    // the same.
    return lhs.element_type() == rhs.element_type();
  }

  if (!rhs.IsArray()) {
    return false;
  }

  if (!ignore_element_type_) {
    if ((ignore_fp_precision_ &&
         !ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) ||
        (!ignore_fp_precision_ && !ShapeUtil::SameElementType(lhs, rhs))) {
      VLOG(3) << "CompareShapes: lhs element type != rhs element type";
      return false;
    }
  }

  if (!ignore_dimensions_) {
    if (!ShapeUtil::SameDimensions(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs dimensions != rhs dimensions";
      return false;
    }
  } else {
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
  }

  if (!ignore_layout_) {
    if (lhs.layout().format() != rhs.layout().format()) {
      VLOG(3) << "CompareShapes: lhs layout format != rhs layout format";
      return false;
    }
    if (LayoutUtil::IsDenseArray(lhs)) {
      Layout::Equal equal;
      if (ignore_tiles_in_layout_) {
        equal.IgnoreTiles();
      }
      if (ignore_element_size_in_layout_) {
        equal.IgnoreElementSize();
      }
      if (ignore_memory_space_in_layout_) {
        equal.IgnoreMemorySpace();
      }
      if (!equal(lhs.layout(), rhs.layout())) {
        VLOG(3) << "CompareShapes: lhs layout != rhs layout";
        return false;
      }
    }
  }

  if (!ignore_dynamic_dimension_) {
    for (int i = 0; i < lhs.rank(); ++i) {
      if (lhs.is_dynamic_dimension(i) != rhs.is_dynamic_dimension(i)) {
        VLOG(3)
            << "CompareShapes: lhs and rhs have different dynamic dimensions.";
        return false;
      }
    }
  }
  return true;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_6(mht_6_v, 397, "", "./tensorflow/compiler/xla/shape.cc", "operator<<");

  out << shape.ToString(/*print_layout=*/true);
  return out;
}

ProgramShape::ProgramShape(const ProgramShapeProto& program_shape_proto) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_7(mht_7_v, 405, "", "./tensorflow/compiler/xla/shape.cc", "ProgramShape::ProgramShape");

  for (const ShapeProto& shape_proto : program_shape_proto.parameters()) {
    *add_parameters() = Shape(shape_proto);
  }
  *mutable_result() = Shape(program_shape_proto.result());
  for (const std::string& name : program_shape_proto.parameter_names()) {
    add_parameter_names(name);
  }
}

ProgramShapeProto ProgramShape::ToProto() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_8(mht_8_v, 418, "", "./tensorflow/compiler/xla/shape.cc", "ProgramShape::ToProto");

  ProgramShapeProto proto;
  for (const Shape& shape : parameters()) {
    *proto.add_parameters() = shape.ToProto();
  }
  *proto.mutable_result() = result().ToProto();
  for (const std::string& name : parameter_names()) {
    proto.add_parameter_names(name);
  }
  return proto;
}

std::string ProgramShape::ToString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_9(mht_9_v, 433, "", "./tensorflow/compiler/xla/shape.cc", "ProgramShape::ToString");

  std::vector<std::string> parameter_strings(parameters_size());
  for (int i = 0; i < parameters_size(); ++i) {
    parameter_strings[i] = absl::StrCat(
        i < parameter_names_size() ? parameter_names(i) : "(unknown)", ": ",
        ShapeUtil::HumanString(parameters(i)));
  }
  return absl::StrCat("(", absl::StrJoin(parameter_strings, ", "), ") -> ",
                      ShapeUtil::HumanString(result()));
}

std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTcc mht_10(mht_10_v, 447, "", "./tensorflow/compiler/xla/shape.cc", "operator<<");

  out << program_shape.ToString() << "\n";
  return out;
}

}  // namespace xla
