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
class MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc() {
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

#include "tensorflow/compiler/xla/layout_util.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

// Internal helper for GetDefaultLayoutForShape and SetToDefaultLayout. Sets
// minor_to_major to the value that represents the default layout.
template <typename T>
void SetDefaultLayoutToContainer(T* minor_to_major) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/layout_util.cc", "SetDefaultLayoutToContainer");

  // The default XLA layout is major-to-minor (dim 0 is major).
  // For more information on XLA layouts, see:
  // https://www.tensorflow.org/performance/xla/shapes
  const int64_t size = minor_to_major->size();
  for (int64_t i = 0; i < size; ++i) {
    (*minor_to_major)[i] = size - 1 - i;
  }
}

}  // namespace

/* static */ Layout LayoutUtil::MakeLayout(
    absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
    int64_t element_size_in_bits, int64_t memory_space) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MakeLayout");

  Layout layout;
  layout.set_format(DENSE);
  for (int64_t dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
  }
  for (const Tile& tile : tiles) {
    for (int64_t dim : tile.dimensions()) {
      if (dim < 0 && dim != Tile::kCombineDimension) {
        LOG(FATAL)
            << "Tile dimension size needs to be minimum int64_t value if "
               "it's negative. Value is "
            << dim;
      }
    }
    *layout.add_tiles() = tile;
  }
  layout.set_element_size_in_bits(element_size_in_bits);
  layout.set_memory_space(memory_space);
  return layout;
}

/* static */ Layout LayoutUtil::MakeDescendingLayout(int64_t rank) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MakeDescendingLayout");

  std::vector<int64_t> layout(rank);
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeAscendingLayout(int64_t rank) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MakeAscendingLayout");

  std::vector<int64_t> layout(rank);
  std::iota(layout.begin(), layout.end(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeLayoutFromMajorToMinor(
    absl::Span<const int64_t> major_to_minor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_4(mht_4_v, 276, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MakeLayoutFromMajorToMinor");

  Layout layout;
  layout.set_format(DENSE);
  for (int i = major_to_minor.size() - 1; i >= 0; i--) {
    layout.add_minor_to_major(major_to_minor[i]);
  }
  return layout;
}

namespace {

// Internal helper that creates a default layout for an array of the given rank.
Layout CreateDefaultLayoutForRank(int64_t rank) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_5(mht_5_v, 291, "", "./tensorflow/compiler/xla/layout_util.cc", "CreateDefaultLayoutForRank");

  Layout layout;
  layout.set_format(DENSE);
  auto* minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->resize(rank, 0);
  SetDefaultLayoutToContainer(minor_to_major);
  return layout;
}

}  // namespace

/* static */ Layout LayoutUtil::GetDefaultLayoutForShape(const Shape& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_6(mht_6_v, 305, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetDefaultLayoutForShape");

  if (shape.IsOpaque() || shape.IsToken()) {
    // Opaque and token types have empty layouts.
    return Layout();
  }

  // A Layout proto corresponds to a single array, not a tuple.
  CHECK(shape.IsArray());
  return CreateDefaultLayoutForRank(shape.dimensions_size());
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForRank(int64_t rank) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_7(mht_7_v, 319, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetDefaultLayoutForRank");

  return CreateDefaultLayoutForRank(rank);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR2() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_8(mht_8_v, 326, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetDefaultLayoutForR2");

  return CreateDefaultLayoutForRank(2);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR3() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_9(mht_9_v, 333, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetDefaultLayoutForR3");

  return CreateDefaultLayoutForRank(3);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR4() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_10(mht_10_v, 340, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetDefaultLayoutForR4");

  return CreateDefaultLayoutForRank(4);
}

/* static */ void LayoutUtil::SetToDefaultLayout(Shape* shape) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_11(mht_11_v, 347, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::SetToDefaultLayout");

  if (shape->IsTuple()) {
    // Tuple shape.
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      SetToDefaultLayout(&element_shape);
    }
    shape->clear_layout();
  } else if (shape->IsArray()) {
    shape->mutable_layout()->set_format(DENSE);
    auto* minor_to_major = shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major->resize(shape->dimensions_size(), 0);
    SetDefaultLayoutToContainer(minor_to_major);
  } else {
    // Opaque, token types etc. have no layout.
    shape->clear_layout();
  }
}

/* static */ Shape LayoutUtil::GetWithDefaultLayout(const Shape& shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_12(mht_12_v, 368, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::GetWithDefaultLayout");

  Shape copy(shape);
  LayoutUtil::SetToDefaultLayout(&copy);
  return copy;
}

/* static */ void LayoutUtil::SetToDefaultLayout(ProgramShape* program_shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_13(mht_13_v, 377, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::SetToDefaultLayout");

  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::SetToDefaultLayout(&parameter_shape);
  }
  LayoutUtil::SetToDefaultLayout(program_shape->mutable_result());
}

/* static */ Status LayoutUtil::ValidateLayoutInShape(
    const Shape& shape, bool allow_missing_layouts) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_14(mht_14_v, 388, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::ValidateLayoutInShape");

  if (shape.IsTuple()) {
    // Tuple shape.
    if (shape.has_layout()) {
      return InvalidArgument("tuple should not have a layout field");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(
          ValidateLayoutInShape(element_shape, allow_missing_layouts));
    }
    return Status::OK();
  } else if (shape.IsArray()) {
    if (!shape.has_layout()) {
      if (allow_missing_layouts) {
        return Status::OK();
      }
      return InvalidArgument("shape %s does not have a layout",
                             ShapeUtil::HumanString(shape));
    }
    return ValidateLayoutForShape(shape.layout(), shape);
  } else {
    // Token, opaque, etc. shape.
    if (shape.has_layout()) {
      return InvalidArgument(
          "shape of primitive type %s should not have a layout",
          PrimitiveType_Name(shape.element_type()));
    }
    return Status::OK();
  }
}

/* static */ Status LayoutUtil::ValidateLayoutForShape(const Layout& layout,
                                                       const Shape& shape) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_15(mht_15_v, 423, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::ValidateLayoutForShape");

  if (shape.IsTuple()) {
    return InvalidArgument("a single Layout is not valid for tuple shapes");
  }

  if (!shape.IsArray()) {
    if (layout.minor_to_major_size() != 0) {
      return InvalidArgument(
          "shape of primitive type %s should not have a non-trivial layout",
          PrimitiveType_Name(shape.element_type()));
    }
    return Status::OK();
  }

  if (layout.format() == INVALID_FORMAT || !Format_IsValid(layout.format())) {
    return InvalidArgument("Layout has an invalid format (%d)",
                           layout.format());
  }

  if (layout.format() == DENSE) {
    if (layout.minor_to_major_size() != shape.rank()) {
      return InvalidArgument(
          "layout minor_to_major field contains %d elements, "
          "but shape is rank %d: {%s}; shape: %s",
          layout.minor_to_major_size(), shape.rank(),
          absl::StrJoin(layout.minor_to_major(), ", "),
          shape.ShortDebugString());
    }

    std::vector<bool> dimensions_in_layout(shape.rank(), false);
    for (int64_t i = 0; i < shape.rank(); ++i) {
      int64_t dim = layout.minor_to_major(i);
      if (dim < 0 || dim >= shape.rank()) {
        return InvalidArgument(
            "layout minor_to_major field has out-of-bounds value: %s",
            HumanString(layout));
      }
      if (dimensions_in_layout[dim]) {
        return InvalidArgument(
            "layout minor_to_major field has duplicate values: {%s}",
            HumanString(layout));
      }
      dimensions_in_layout[dim] = true;
    }
  } else {
    if (layout.tiles_size() != 0) {
      return InvalidArgument("Only dense layouts can be tiled.");
    }
  }

  return Status::OK();
}

/* static */ void LayoutUtil::ClearLayout(Shape* shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_16(mht_16_v, 479, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::ClearLayout");

  shape->clear_layout();
  for (auto& element_shape : *shape->mutable_tuple_shapes()) {
    ClearLayout(&element_shape);
  }
}

/* static */ void LayoutUtil::ClearLayout(ProgramShape* program_shape) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_17(mht_17_v, 489, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::ClearLayout");

  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::ClearLayout(&parameter_shape);
  }
  LayoutUtil::ClearLayout(program_shape->mutable_result());
}

/* static */ bool LayoutUtil::IsDenseArray(const Shape& shape) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_18(mht_18_v, 499, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::IsDenseArray");

  return shape.IsArray() && shape.has_layout() && IsDense(shape.layout());
}

/* static */ bool LayoutUtil::IsDense(const Layout& layout) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_19(mht_19_v, 506, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::IsDense");

  return layout.format() == DENSE;
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Minor(const Layout& layout) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_20(mht_20_v, 513, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::IsMonotonicWithDim0Minor");

  CHECK(layout.format() == DENSE);
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Major(const Layout& layout) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_21(mht_21_v, 522, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::IsMonotonicWithDim0Major");

  CHECK(layout.format() == DENSE);
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64_t>());
}

/* static */ bool LayoutUtil::HasLayout(const Shape& shape) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_22(mht_22_v, 531, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::HasLayout");

  if (shape.IsTuple()) {
    // Tuple shape: all subshapes must have a layout.
    return absl::c_all_of(shape.tuple_shapes(),
                          [](const Shape& s) { return HasLayout(s); });
  } else if (!shape.IsArray()) {
    // Opaque, token types etc. ignore layout.
    return true;
  }
  return shape.has_layout() && shape.layout().format() != INVALID_FORMAT;
}

/* static */ bool LayoutUtil::HasLayout(const ProgramShape& program_shape) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_23(mht_23_v, 546, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::HasLayout");

  for (auto& parameter_shape : program_shape.parameters()) {
    if (!LayoutUtil::HasLayout(parameter_shape)) {
      return false;
    }
  }
  return LayoutUtil::HasLayout(program_shape.result());
}

/* static */ bool LayoutUtil::Equal(const Layout& lhs, const Layout& rhs) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_24(mht_24_v, 558, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::Equal");

  return lhs == rhs;
}

/* static */ absl::Span<const int64_t> LayoutUtil::MinorToMajor(
    const Shape& shape) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_25(mht_25_v, 566, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MinorToMajor");

  CHECK(IsDenseArray(shape));
  return shape.layout().minor_to_major();
}

/* static */ absl::Span<const int64_t> LayoutUtil::MinorToMajor(
    const Layout& layout) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_26(mht_26_v, 575, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MinorToMajor");

  CHECK(layout.format() == DENSE);
  return layout.minor_to_major();
}

/* static */ int64_t LayoutUtil::Major(const Layout& layout,
                                       int64_t physical_dimension_number) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_27(mht_27_v, 584, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::Major");

  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return Minor(layout,
               layout.minor_to_major_size() - 1 - physical_dimension_number);
}

/* static */ int64_t LayoutUtil::Minor(const Layout& layout,
                                       int64_t physical_dimension_number) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_28(mht_28_v, 595, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::Minor");

  CHECK_EQ(layout.format(), DENSE);
  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return layout.minor_to_major(physical_dimension_number);
}

/* static */ std::vector<int64_t> LayoutUtil::MakeLogicalToPhysical(
    const Layout& layout) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_29(mht_29_v, 606, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MakeLogicalToPhysical");

  std::vector<int64_t> logical_to_physical(layout.minor_to_major_size());
  for (int64_t physical = 0, end = logical_to_physical.size(); physical < end;
       ++physical) {
    const int64_t logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}

/* static */ std::string LayoutUtil::HumanString(const Layout& layout) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_30(mht_30_v, 619, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::HumanString");

  return layout.ToString();
}

namespace {

// Internal helper for recursively copying layouts.
Status CopyLayoutInternal(const Shape& src, Shape* dst) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_31(mht_31_v, 629, "", "./tensorflow/compiler/xla/layout_util.cc", "CopyLayoutInternal");

  if (src.IsTuple() != dst->IsTuple()) {
    return InvalidArgument(
        "cannot copy layout from shape: shape structure differs");
  }
  if (src.IsTuple()) {
    if (ShapeUtil::TupleElementCount(src) !=
        ShapeUtil::TupleElementCount(*dst)) {
      return InvalidArgument(
          "cannot copy layout from shape: tuple element count differs");
    }
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(src); ++i) {
      TF_RETURN_IF_ERROR(CopyLayoutInternal(src.tuple_shapes(i),
                                            dst->mutable_tuple_shapes(i)));
    }
  } else {
    if (src.has_layout()) {
      if (src.rank() != dst->rank()) {
        return InvalidArgument("cannot copy layout from shape: ranks differs");
      }
      TF_RETURN_IF_ERROR(
          LayoutUtil::ValidateLayoutForShape(src.layout(), *dst));
      *dst->mutable_layout() = src.layout();
    } else {
      dst->clear_layout();
    }
  }
  return Status::OK();
}

}  // namespace

/* static */
Status LayoutUtil::CopyLayoutBetweenShapes(const Shape& src, Shape* dst) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_32(mht_32_v, 665, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::CopyLayoutBetweenShapes");

  return CopyLayoutInternal(src, dst);
}

/* static */ bool LayoutUtil::LayoutsInShapesEqual(const Shape& lhs,
                                                   const Shape& rhs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_33(mht_33_v, 673, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::LayoutsInShapesEqual");

  if (lhs.IsTuple()) {
    if (!rhs.IsTuple() || ShapeUtil::TupleElementCount(lhs) !=
                              ShapeUtil::TupleElementCount(rhs)) {
      return false;
    }
    for (int i = 0; i < ShapeUtil::TupleElementCount(lhs); ++i) {
      if (!LayoutsInShapesEqual(lhs.tuple_shapes(i), rhs.tuple_shapes(i))) {
        return false;
      }
    }
    return true;
  } else if (lhs.IsArray()) {
    return lhs.rank() == rhs.rank() &&
           LayoutUtil::Equal(lhs.layout(), rhs.layout());
  } else {
    // Layouts of non-array and non-tuple shapes is ignored.
    return true;
  }
}

/* static */ bool LayoutUtil::AreDimensionsConsecutive(
    const Layout& layout, absl::Span<const int64_t> dims) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_34(mht_34_v, 698, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::AreDimensionsConsecutive");

  CHECK(IsDense(layout));
  absl::InlinedVector<int64_t, 8> positions_in_layout;
  for (int64_t dim : dims) {
    positions_in_layout.push_back(
        PositionInContainer(layout.minor_to_major(), dim));
  }
  absl::c_sort(positions_in_layout);
  for (size_t i = 1; i < positions_in_layout.size(); ++i) {
    if (1 != positions_in_layout[i] - positions_in_layout[i - 1]) {
      return false;
    }
  }
  return true;
}

/*static*/ Layout LayoutUtil::MoveDimToMajor(const Layout& layout,
                                             int64_t dim) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_35(mht_35_v, 718, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::MoveDimToMajor");

  if (dim == MinorToMajor(layout).back()) return layout;
  Layout ret = layout;
  ret.clear_minor_to_major();
  for (auto d : MinorToMajor(layout)) {
    if (d != dim) {
      ret.add_minor_to_major(d);
    }
  }
  ret.add_minor_to_major(dim);
  return ret;
}

/*static*/ int64_t LayoutUtil::LinearIndex(const Shape& shape,
                                           absl::Span<const int64_t> indices) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayout_utilDTcc mht_36(mht_36_v, 735, "", "./tensorflow/compiler/xla/layout_util.cc", "LayoutUtil::LinearIndex");

  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  const int rank = shape.rank();
  CHECK_EQ(rank, indices.size());

  if (rank == 0) {
    return 0;
  }
  if (rank == 1) {
    return indices[0];
  }

  Tile tile = {};
  if (!shape.layout().tiles().empty()) {
    tile = shape.layout().tiles()[0];
  }

  int64_t linear_index = 0;
  int64_t tile_multiplier = 1;
  // Initialize to number of elements in a tile.
  for (int64_t i : tile.dimensions()) {
    tile_multiplier *= i;
  }
  int64_t within_tile_multiplier = 1;

  // We only look at the top-level tile.
  for (int64_t minor = 0; minor < rank; minor++) {
    int64_t logical_dim = Minor(shape.layout(), minor);
    int64_t shape_dim_size = shape.dimensions(logical_dim);
    int64_t index = indices[logical_dim];

    if (minor < tile.dimensions().size()) {
      int64_t tile_dim_size =
          tile.dimensions()[tile.dimensions().size() - 1 - minor];
      linear_index += tile_multiplier * (index / tile_dim_size) +
                      within_tile_multiplier * (index % tile_dim_size);
      tile_multiplier *= CeilOfRatio(shape_dim_size, tile_dim_size);
      within_tile_multiplier *= tile_dim_size;
    } else {
      linear_index += index * tile_multiplier;
      tile_multiplier *= shape_dim_size;
    }
  }
  return linear_index;
}

}  // namespace xla
