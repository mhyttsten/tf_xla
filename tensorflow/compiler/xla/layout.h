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

#ifndef TENSORFLOW_COMPILER_XLA_LAYOUT_H_
#define TENSORFLOW_COMPILER_XLA_LAYOUT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh() {
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
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Describes a tile used in tiling-based layout. Refer to
// g3doc/third_party/tensorflow/compiler/xla/g3doc/tiled_layout.md for
// details.
class Tile {
 public:
  Tile() = default;
  explicit Tile(absl::Span<const int64_t> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/layout.h", "Tile");
}

  // De/Serialize a Tile to and from a TileProto.
  static Tile CreateFromProto(const TileProto& tile_proto) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/layout.h", "CreateFromProto");

    return Tile(tile_proto.dimensions());
  }
  TileProto ToProto() const;

  bool operator==(const Tile& other) const {
    return dimensions() == other.dimensions();
  }
  bool operator!=(const Tile& other) const { return !(*this == other); }

  std::string ToString() const;

  // Returns the bound of the tile in the given dimension index.
  int64_t dimension(int i) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_2(mht_2_v, 228, "", "./tensorflow/compiler/xla/layout.h", "dimension");
 return dimensions_.at(i); }

  // Returns the dimensions of the tile.
  absl::Span<const int64_t> dimensions() const { return dimensions_; }

  Tile& add_dimensions(int64_t value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/layout.h", "add_dimensions");

    dimensions_.push_back(value);
    return *this;
  }

  Tile& clear_dimensions() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_4(mht_4_v, 244, "", "./tensorflow/compiler/xla/layout.h", "clear_dimensions");

    dimensions_.clear();
    return *this;
  }

  // This dimension size means the corresponding dimension in the shape is
  // combined with the next minor dimension before tiling is applied.
  static constexpr int64_t kCombineDimension =
      std::numeric_limits<int64_t>::min();

  template <typename H>
  friend H AbslHashValue(H h, const Tile& t) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_5(mht_5_v, 258, "", "./tensorflow/compiler/xla/layout.h", "AbslHashValue");

    return H::combine(std::move(h), t.dimensions_);
  }

 private:
  // The bounds of the tile.
  absl::InlinedVector<int64_t, 2> dimensions_;
};

class Layout {
 public:
  Layout() = default;

  // Constructs a dense layout with the given minor-to-major order.
  explicit Layout(absl::Span<const int64_t> minor_to_major)
      : format_(DENSE),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_6(mht_6_v, 277, "", "./tensorflow/compiler/xla/layout.h", "Layout");
}

  // Constructs a dense tiled layout with the given minor-to-major order and
  // tiles.
  Layout(absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
         int64_t element_size_in_bits = 0, int64_t memory_space = 0)
      : format_(DENSE),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()),
        tiles_(tiles.begin(), tiles.end()),
        element_size_in_bits_(element_size_in_bits),
        memory_space_(memory_space) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_7(mht_7_v, 290, "", "./tensorflow/compiler/xla/layout.h", "Layout");
}

  // Construct a shape from a LayoutProto.
  static Layout CreateFromProto(const LayoutProto& proto);

  // Returns a LayoutProto representation of the Layout.
  LayoutProto ToProto() const;

  // Returns a human-readable string that represents this layout.
  std::string ToString() const;

  // Equal is a configurable functor to check the equality of two layouts.
  //
  // Examples:
  //
  // - Comparing two layouts ignoring their difference in tiles:
  //   Equal().IgnoreTiles()(layout1, layout2);
  //
  // - Comparing two layouts ignoring their difference in tiles and element
  //   size:
  //   Equal().IgnoreTiles().IgnoreElementSize()(layout1, layout2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Layout& lhs, const Layout& rhs);

    Equal& IgnoreTiles() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_8(mht_8_v, 320, "", "./tensorflow/compiler/xla/layout.h", "IgnoreTiles");

      ignore_tiles_ = true;
      return *this;
    }

    Equal& IgnoreElementSize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_9(mht_9_v, 328, "", "./tensorflow/compiler/xla/layout.h", "IgnoreElementSize");

      ignore_element_size_ = true;
      return *this;
    }

    Equal& MinorToMajorOnly() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_10(mht_10_v, 336, "", "./tensorflow/compiler/xla/layout.h", "MinorToMajorOnly");

      ignore_tiles_ = true;
      ignore_element_size_ = true;
      ignore_memory_space_ = true;
      return *this;
    }

    Equal& IgnoreMemorySpace() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_11(mht_11_v, 346, "", "./tensorflow/compiler/xla/layout.h", "IgnoreMemorySpace");

      ignore_memory_space_ = true;
      return *this;
    }

   private:
    bool ignore_tiles_ = false;
    bool ignore_element_size_ = false;
    bool ignore_memory_space_ = false;
  };

  bool operator==(const Layout& other) const;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  // The following methods mirror the protobuf generated code interface for the
  // message LayoutProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  //
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the format.
  Format format() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_12(mht_12_v, 371, "", "./tensorflow/compiler/xla/layout.h", "format");
 return format_; }
  Layout& set_format(Format value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_13(mht_13_v, 375, "", "./tensorflow/compiler/xla/layout.h", "set_format");

    format_ = value;
    return *this;
  }

  // Methods for accessing the minor-to-major array.
  int minor_to_major_size() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_14(mht_14_v, 384, "", "./tensorflow/compiler/xla/layout.h", "minor_to_major_size");
 return minor_to_major_.size(); }
  int64_t minor_to_major(int index) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_15(mht_15_v, 388, "", "./tensorflow/compiler/xla/layout.h", "minor_to_major");
 return minor_to_major_.at(index); }
  Layout& set_minor_to_major(int index, int64_t value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_16(mht_16_v, 392, "", "./tensorflow/compiler/xla/layout.h", "set_minor_to_major");

    minor_to_major_.at(index) = value;
    return *this;
  }
  Layout& add_minor_to_major(int64_t value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_17(mht_17_v, 399, "", "./tensorflow/compiler/xla/layout.h", "add_minor_to_major");

    minor_to_major_.push_back(value);
    return *this;
  }
  Layout& clear_minor_to_major() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_18(mht_18_v, 406, "", "./tensorflow/compiler/xla/layout.h", "clear_minor_to_major");

    minor_to_major_.clear();
    return *this;
  }
  absl::Span<const int64_t> minor_to_major() const { return minor_to_major_; }
  absl::InlinedVector<int64_t, 6>* mutable_minor_to_major() {
    return &minor_to_major_;
  }

  // Methods for accessing the tile field.
  int tiles_size() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_19(mht_19_v, 419, "", "./tensorflow/compiler/xla/layout.h", "tiles_size");
 return tiles_.size(); }
  const Tile& tiles(int index) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_20(mht_20_v, 423, "", "./tensorflow/compiler/xla/layout.h", "tiles");
 return tiles_.at(index); }
  Tile* mutable_tiles(int index) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_21(mht_21_v, 427, "", "./tensorflow/compiler/xla/layout.h", "mutable_tiles");
 return &tiles_.at(index); }
  Tile* add_tiles() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_22(mht_22_v, 431, "", "./tensorflow/compiler/xla/layout.h", "add_tiles");

    tiles_.push_back(Tile());
    return &tiles_.back();
  }
  Layout& clear_tiles() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_23(mht_23_v, 438, "", "./tensorflow/compiler/xla/layout.h", "clear_tiles");

    tiles_.clear();
    return *this;
  }
  absl::Span<const Tile> tiles() const { return tiles_; }
  absl::InlinedVector<Tile, 2>* mutable_tiles() { return &tiles_; }

  int64_t element_size_in_bits() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_24(mht_24_v, 448, "", "./tensorflow/compiler/xla/layout.h", "element_size_in_bits");
 return element_size_in_bits_; }
  Layout& set_element_size_in_bits(int64_t value) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_25(mht_25_v, 452, "", "./tensorflow/compiler/xla/layout.h", "set_element_size_in_bits");

    element_size_in_bits_ = value;
    return *this;
  }
  static constexpr int64_t kDefaultMemorySpace = 0;
  static constexpr int64_t kGenericFastMemorySpace = 1;
  int64_t memory_space() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_26(mht_26_v, 461, "", "./tensorflow/compiler/xla/layout.h", "memory_space");
 return memory_space_; }
  Layout& set_memory_space(int64_t value) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_27(mht_27_v, 465, "", "./tensorflow/compiler/xla/layout.h", "set_memory_space");

    memory_space_ = value;
    return *this;
  }

  void Swap(Layout* other) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_28(mht_28_v, 473, "", "./tensorflow/compiler/xla/layout.h", "Swap");

    using std::swap;
    swap(*this, *other);
  }

  void Clear() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_29(mht_29_v, 481, "", "./tensorflow/compiler/xla/layout.h", "Clear");

    *this = Layout();
    format_ = INVALID_FORMAT;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Layout& l) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTh mht_30(mht_30_v, 490, "", "./tensorflow/compiler/xla/layout.h", "AbslHashValue");

    return H::combine(std::move(h), l.format_, l.minor_to_major_, l.tiles_,
                      l.element_size_in_bits_, l.memory_space_);
  }

 private:
  // The format of this layout.
  Format format_ = INVALID_FORMAT;

  // A map from physical dimension numbers to logical dimension numbers.
  // The first element is the most minor physical dimension (fastest varying
  // index) and the last the most major (slowest varying index). The contents of
  // the vector are the indices of the *logical* dimensions in the shape.
  //
  // For example, in shape f32[8,100,100,3]{3,0,2,1}, the logical dimensions
  // are [8,100,100,3] and minor_to_major_ is {3,0,2,1}.
  // So, the most minor physical dimension is [8,100,100,3][3], which is size 3.
  // The second most minor is [8,100,100,3][0], which is size 8.
  // The third most minor is [8,100,100,3][2], which is size 100.
  // And the major dim is [8,100,100,3][1], which is size 100.
  absl::InlinedVector<int64_t, 6> minor_to_major_;

  // The tiles used in tiling-based layout.
  absl::InlinedVector<Tile, 2> tiles_;

  // The number of bits used to store an individual array element.
  int64_t element_size_in_bits_ = 0;

  // The assigned memory space.
  int64_t memory_space_ = 0;
};

std::ostream& operator<<(std::ostream& out, const Tile& Tile);
std::ostream& operator<<(std::ostream& out, const Layout& layout);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LAYOUT_H_
