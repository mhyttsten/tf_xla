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

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh() {
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
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A shape describes the number of dimensions in a array, the bounds of each
// dimension, and the primitive component type. For tuples, shape describes the
// structure (number of elements and nesting).
class Shape {
 public:
  Shape() = default;

  // Construct a shape from a ShapeProto.
  explicit Shape(const ShapeProto& shape_proto);

  Shape(PrimitiveType element_type, absl::Span<const int64_t> dimensions,
        absl::Span<const bool> dynamic_dimensions,
        std::vector<Shape> tuple_shapes)
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()),
        dynamic_dimensions_(dynamic_dimensions.begin(),
                            dynamic_dimensions.end()),
        tuple_shapes_(std::move(tuple_shapes)) {}

  // Returns a ShapeProto representation of the Shape.
  ShapeProto ToProto() const;

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  std::string ToString(bool print_layout = false) const;

  // Returns the rank (number of dimensions) of the given shape. Shape must be
  // an array.
  int64_t rank() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_0(mht_0_v, 229, "", "./tensorflow/compiler/xla/shape.h", "rank");

    DCHECK(IsArray()) << "Non-arrays do not have a rank, shape: " << ToString();
    return dimensions_.size();
  }

  // Returns whether the shape is of the specified type (array, tuple, etc).
  bool IsArray() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_1(mht_1_v, 238, "", "./tensorflow/compiler/xla/shape.h", "IsArray");
 return primitive_util::IsArrayType(element_type()); }
  bool IsTuple() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_2(mht_2_v, 242, "", "./tensorflow/compiler/xla/shape.h", "IsTuple");
 return element_type() == TUPLE; }
  bool IsToken() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_3(mht_3_v, 246, "", "./tensorflow/compiler/xla/shape.h", "IsToken");
 return element_type() == TOKEN; }
  bool IsOpaque() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_4(mht_4_v, 250, "", "./tensorflow/compiler/xla/shape.h", "IsOpaque");
 return element_type() == OPAQUE_TYPE; }

  // Returns whether all elements in the shape are integer.
  // A nested tuple of integers is considered as integer.
  bool IsInteger() const;

  // Returns true if no array dimension in the shape is dynamically sized. Tuple
  // shapes are traversed recursively.
  bool is_static() const;

  bool is_dynamic() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_5(mht_5_v, 263, "", "./tensorflow/compiler/xla/shape.h", "is_dynamic");
 return !is_static(); }

  // Returns true if the given dimension is dynamically-sized.
  bool is_dynamic_dimension(int dimension) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_6(mht_6_v, 269, "", "./tensorflow/compiler/xla/shape.h", "is_dynamic_dimension");

    return dynamic_dimensions_.at(dimension);
  }

  // Sets whether or not the given dimension is dynamically-sized.
  void set_dynamic_dimension(int dimension, bool is_dynamic) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_7(mht_7_v, 277, "", "./tensorflow/compiler/xla/shape.h", "set_dynamic_dimension");

    dynamic_dimensions_[dimension] = is_dynamic;
  }

  absl::Span<const bool> dynamic_dimensions() const {
    return dynamic_dimensions_;
  }

  absl::Span<bool> mutable_dynamic_dimensions() {
    return absl::MakeSpan(dynamic_dimensions_);
  }

  // Add dimension_upper_bound().

  // Removes the given dimension form the shape. Layout, if it exists, is
  // adjusted to match the modified shape.
  void DeleteDimension(int64_t dim_to_delete);

  // The following methods mirror the protobuf generated code interface for the
  // message ShapeProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the primitive type.
  PrimitiveType element_type() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_8(mht_8_v, 305, "", "./tensorflow/compiler/xla/shape.h", "element_type");
 return element_type_; }
  void set_element_type(PrimitiveType value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/shape.h", "set_element_type");
 element_type_ = value; }

  // Methods for accessing the dimensions array.
  int dimensions_size() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_10(mht_10_v, 315, "", "./tensorflow/compiler/xla/shape.h", "dimensions_size");
 return dimensions_.size(); }
  int64_t dimensions(int index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_11(mht_11_v, 319, "", "./tensorflow/compiler/xla/shape.h", "dimensions");
 return dimensions_.at(index); }
  void set_dimensions(int index, int64_t value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_12(mht_12_v, 323, "", "./tensorflow/compiler/xla/shape.h", "set_dimensions");

    dimensions_.at(index) = value;
  }
  void add_dimensions(int64_t value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_13(mht_13_v, 329, "", "./tensorflow/compiler/xla/shape.h", "add_dimensions");

    dimensions_.push_back(value);
    dynamic_dimensions_.push_back(false);
  }
  void clear_dimensions() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_14(mht_14_v, 336, "", "./tensorflow/compiler/xla/shape.h", "clear_dimensions");

    dimensions_.clear();
    dynamic_dimensions_.clear();
  }
  absl::Span<const int64_t> dimensions() const { return dimensions_; }
  absl::Span<int64_t> mutable_dimensions() {
    return absl::MakeSpan(dimensions_);
  }

  // Methods for accessing the tuple subshapes. This field only non-empty for
  // tuple shapes.
  int tuple_shapes_size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_15(mht_15_v, 350, "", "./tensorflow/compiler/xla/shape.h", "tuple_shapes_size");
 return tuple_shapes_.size(); }
  const Shape& tuple_shapes(int index) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_16(mht_16_v, 354, "", "./tensorflow/compiler/xla/shape.h", "tuple_shapes");
 return tuple_shapes_.at(index); }
  Shape* mutable_tuple_shapes(int index) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_17(mht_17_v, 358, "", "./tensorflow/compiler/xla/shape.h", "mutable_tuple_shapes");
 return &tuple_shapes_.at(index); }
  Shape* add_tuple_shapes() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_18(mht_18_v, 362, "", "./tensorflow/compiler/xla/shape.h", "add_tuple_shapes");

    tuple_shapes_.push_back(Shape());
    return &tuple_shapes_.back();
  }
  void clear_tuple_shapes() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_19(mht_19_v, 369, "", "./tensorflow/compiler/xla/shape.h", "clear_tuple_shapes");
 tuple_shapes_.clear(); }
  const std::vector<Shape>& tuple_shapes() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_20(mht_20_v, 373, "", "./tensorflow/compiler/xla/shape.h", "tuple_shapes");
 return tuple_shapes_; }
  std::vector<Shape>* mutable_tuple_shapes() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_21(mht_21_v, 377, "", "./tensorflow/compiler/xla/shape.h", "mutable_tuple_shapes");
 return &tuple_shapes_; }

  // Methods for accessing the layout field.
  bool has_layout() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_22(mht_22_v, 383, "", "./tensorflow/compiler/xla/shape.h", "has_layout");
 return layout_.format() != INVALID_FORMAT; }
  const Layout& layout() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_23(mht_23_v, 387, "", "./tensorflow/compiler/xla/shape.h", "layout");
 return layout_; }
  Layout* mutable_layout() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_24(mht_24_v, 391, "", "./tensorflow/compiler/xla/shape.h", "mutable_layout");
 return &layout_; }
  void clear_layout() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_25(mht_25_v, 395, "", "./tensorflow/compiler/xla/shape.h", "clear_layout");
 layout_.Clear(); }

  // Recursively clear dynamic dimension of a shape.
  void clear_dynamic_dimensions() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_26(mht_26_v, 401, "", "./tensorflow/compiler/xla/shape.h", "clear_dynamic_dimensions");

    if (!IsTuple()) {
      for (int64_t i = 0; i < dynamic_dimensions_.size(); ++i) {
        dynamic_dimensions_[i] = false;
      }
      return;
    }
    for (auto& subshape : tuple_shapes_) {
      subshape.clear_dynamic_dimensions();
    }
  }

  void Swap(Shape* other) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_27(mht_27_v, 416, "", "./tensorflow/compiler/xla/shape.h", "Swap");

    using std::swap;
    swap(*this, *other);
  }

  void Clear() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_28(mht_28_v, 424, "", "./tensorflow/compiler/xla/shape.h", "Clear");

    element_type_ = PRIMITIVE_TYPE_INVALID;
    clear_dimensions();
    tuple_shapes_.clear();
    clear_layout();
  }

  std::string SerializeAsString() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_29(mht_29_v, 434, "", "./tensorflow/compiler/xla/shape.h", "SerializeAsString");

    return ToProto().SerializeAsString();
  }
  std::string ShortDebugString() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_30(mht_30_v, 440, "", "./tensorflow/compiler/xla/shape.h", "ShortDebugString");
 return ToProto().ShortDebugString(); }
  std::string DebugString() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_31(mht_31_v, 444, "", "./tensorflow/compiler/xla/shape.h", "DebugString");
 return ToProto().DebugString(); }

  // Equal is a configurable functor to check the equality of two shapes.
  //
  // Examples:
  //
  // - Comparing two shapes ignoring their layout difference:
  //   Equal().IgnoreLayout()(shape1, shape2);
  //
  // - Comparing two shapes ignoring their layout and element type difference:
  //   Equal().IgnoreLayout().IgnoreElementType()(shape1, shape2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Shape& lhs, const Shape& rhs);

    Equal& IgnoreLayout() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_32(mht_32_v, 464, "", "./tensorflow/compiler/xla/shape.h", "IgnoreLayout");

      ignore_layout_ = true;
      return *this;
    }
    Equal& IgnoreTilesInLayout() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_33(mht_33_v, 471, "", "./tensorflow/compiler/xla/shape.h", "IgnoreTilesInLayout");

      ignore_tiles_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementSizeInLayout() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_34(mht_34_v, 478, "", "./tensorflow/compiler/xla/shape.h", "IgnoreElementSizeInLayout");

      ignore_element_size_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreMemorySpaceInLayout() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_35(mht_35_v, 485, "", "./tensorflow/compiler/xla/shape.h", "IgnoreMemorySpaceInLayout");

      ignore_memory_space_in_layout_ = true;
      return *this;
    }
    Equal& MinorToMajorOnlyInLayout() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_36(mht_36_v, 492, "", "./tensorflow/compiler/xla/shape.h", "MinorToMajorOnlyInLayout");

      ignore_tiles_in_layout_ = true;
      ignore_element_size_in_layout_ = true;
      ignore_memory_space_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementType() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_37(mht_37_v, 501, "", "./tensorflow/compiler/xla/shape.h", "IgnoreElementType");

      ignore_element_type_ = true;
      return *this;
    }
    Equal& IgnoreFpPrecision() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_38(mht_38_v, 508, "", "./tensorflow/compiler/xla/shape.h", "IgnoreFpPrecision");

      ignore_fp_precision_ = true;
      return *this;
    }
    Equal& IgnoreDynamicDimension() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_39(mht_39_v, 515, "", "./tensorflow/compiler/xla/shape.h", "IgnoreDynamicDimension");

      ignore_dynamic_dimension_ = true;
      return *this;
    }
    Equal& IgnoreDimensions() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_40(mht_40_v, 522, "", "./tensorflow/compiler/xla/shape.h", "IgnoreDimensions");

      ignore_dimensions_ = true;
      return *this;
    }

   private:
    bool ignore_layout_ = false;
    bool ignore_tiles_in_layout_ = false;
    bool ignore_element_size_in_layout_ = false;
    bool ignore_memory_space_in_layout_ = false;
    bool ignore_element_type_ = false;
    bool ignore_fp_precision_ = false;
    bool ignore_dynamic_dimension_ = false;
    bool ignore_dimensions_ = false;
  };

  // Test that all fields of the shape are the same, equivalent to Equal().
  bool operator==(const Shape& other) const { return Equal()(*this, other); }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  template <typename H, bool kIsLayoutSensitive = true>
  static H Hash(H h, const Shape& s) {
    if (s.IsTuple()) {
      for (const Shape& subshape : s.tuple_shapes_) {
        h = Shape::Hash<H, kIsLayoutSensitive>(std::move(h), subshape);
      }
      return H::combine(std::move(h), s.tuple_shapes_size());
    }
    h = H::combine(std::move(h), s.element_type_, s.dimensions_,
                   s.dynamic_dimensions_);
    if (kIsLayoutSensitive) {
      h = H::combine(std::move(h), s.layout_);
    }
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& s) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_41(mht_41_v, 562, "", "./tensorflow/compiler/xla/shape.h", "AbslHashValue");

    return Shape::Hash(std::move(h), s);
  }

 private:
  // The element type of this shape (tuple, array, etc).
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;

  // The array bounds of the dimensions. This is nonempty only for array
  // shapes. For a dynamically-sized dimension, the respective value in this
  // vector is an inclusive upper limit of the array bound.
  absl::InlinedVector<int64_t, 6> dimensions_;

  // This vector is the same size as 'dimensions_' and indicates whether the
  // respective dimension is dynamically sized.
  absl::InlinedVector<bool, 6> dynamic_dimensions_;

  // The tuple element subshapes. This is nonempty only for tuple shapes.
  std::vector<Shape> tuple_shapes_;

  // The layout of the shape. Only relevant for arrays.
  Layout layout_;
};

// Shape of the parameters and output of an XLA computation. This is analogous
// to a traditional function signature.
class ProgramShape {
 public:
  ProgramShape() = default;

  // Creates a ProgramShape from a ProgramShapeProto protobuf.
  explicit ProgramShape(const ProgramShapeProto& program_shape_proto);

  // Returns a proto representation of the object.
  ProgramShapeProto ToProto() const;

  std::string ToString() const;

  // The following methods mirror the protobuf generated code interface for the
  // message ProgramShapeProto. This enabled easy migration of this data
  // structure from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing and manipulating the Shape of the parameters.
  int parameters_size() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_42(mht_42_v, 610, "", "./tensorflow/compiler/xla/shape.h", "parameters_size");
 return parameters_.size(); }
  const Shape& parameters(int index) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_43(mht_43_v, 614, "", "./tensorflow/compiler/xla/shape.h", "parameters");
 return parameters_.at(index); }
  Shape* mutable_parameters(int index) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_44(mht_44_v, 618, "", "./tensorflow/compiler/xla/shape.h", "mutable_parameters");
 return &parameters_.at(index); }
  Shape* add_parameters() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_45(mht_45_v, 622, "", "./tensorflow/compiler/xla/shape.h", "add_parameters");

    parameters_.emplace_back();
    return &parameters_.back();
  }
  void clear_parameters() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_46(mht_46_v, 629, "", "./tensorflow/compiler/xla/shape.h", "clear_parameters");
 parameters_.clear(); }
  const std::vector<Shape>& parameters() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_47(mht_47_v, 633, "", "./tensorflow/compiler/xla/shape.h", "parameters");
 return parameters_; }
  std::vector<Shape>* mutable_parameters() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_48(mht_48_v, 637, "", "./tensorflow/compiler/xla/shape.h", "mutable_parameters");
 return &parameters_; }

  // Methods for accessing and manipulating the Shape of the result.
  const Shape& result() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_49(mht_49_v, 643, "", "./tensorflow/compiler/xla/shape.h", "result");
 return result_; }
  Shape* mutable_result() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_50(mht_50_v, 647, "", "./tensorflow/compiler/xla/shape.h", "mutable_result");
 return &result_; }

  // Methods for accessing and manipulating the names of the parameters.
  int parameter_names_size() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_51(mht_51_v, 653, "", "./tensorflow/compiler/xla/shape.h", "parameter_names_size");
 return parameter_names_.size(); }
  const std::string& parameter_names(int index) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_52(mht_52_v, 657, "", "./tensorflow/compiler/xla/shape.h", "parameter_names");

    return parameter_names_.at(index);
  }
  void set_parameter_names(int index, const std::string& value) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_53(mht_53_v, 664, "", "./tensorflow/compiler/xla/shape.h", "set_parameter_names");

    parameter_names_.at(index) = value;
  }
  std::string* mutable_parameter_names(int index) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_54(mht_54_v, 670, "", "./tensorflow/compiler/xla/shape.h", "mutable_parameter_names");

    return &parameter_names_.at(index);
  }
  void add_parameter_names(const std::string& value) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_55(mht_55_v, 677, "", "./tensorflow/compiler/xla/shape.h", "add_parameter_names");

    parameter_names_.push_back(value);
  }
  std::string* add_parameter_names() {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_56(mht_56_v, 683, "", "./tensorflow/compiler/xla/shape.h", "add_parameter_names");

    parameter_names_.push_back("");
    return &parameter_names_.back();
  }
  void clear_parameter_names() {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_57(mht_57_v, 690, "", "./tensorflow/compiler/xla/shape.h", "clear_parameter_names");
 parameter_names_.clear(); }
  const std::vector<std::string>& parameter_names() const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_58(mht_58_v, 694, "", "./tensorflow/compiler/xla/shape.h", "parameter_names");

    return parameter_names_;
  }
  std::vector<std::string>* mutable_parameter_names() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_59(mht_59_v, 700, "", "./tensorflow/compiler/xla/shape.h", "mutable_parameter_names");

    return &parameter_names_;
  }

  std::string ShortDebugString() const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_60(mht_60_v, 707, "", "./tensorflow/compiler/xla/shape.h", "ShortDebugString");
 return ToProto().ShortDebugString(); }
  std::string DebugString() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshapeDTh mht_61(mht_61_v, 711, "", "./tensorflow/compiler/xla/shape.h", "DebugString");
 return ToProto().DebugString(); }

 private:
  // The shapes of the parameters of the computation represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the computation represented by this object.
  std::vector<std::string> parameter_names_;

  // The shape of the result of the computation represented by this object.
  Shape result_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_H_
