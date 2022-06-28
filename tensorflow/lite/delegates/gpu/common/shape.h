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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SHAPE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SHAPE_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh() {
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


#include <stddef.h>
#include <stdint.h>

#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace tflite {
namespace gpu {

enum class Axis {
  UNKNOWN = 0,
  CHANNELS = 1,
  INPUT_CHANNELS = 2,
  OUTPUT_CHANNELS = 3,
  HEIGHT = 4,
  WIDTH = 5,
  BATCH = 6,
  VALUE = 7,
  DEPTH = 8,
};

std::string ToString(Axis t);

// Layout represents axis order.
enum class Layout {
  UNKNOWN = 0,
  SCALAR = 1,
  LINEAR = 2,
  HW = 3,
  CHW = 4,
  HWC = 5,
  OIHW = 6,
  OHWI = 7,
  IHWO = 8,
  IOHW = 9,
  BHWC = 10,
  HWDC = 11,
  BHWDC = 12,
  HWD = 13,
  OHWDI = 14,
};

std::string ToString(Layout l);

// Returns number of axis for the fixed layout.
template <Layout T>
constexpr int Size();

// Returns number of axis for the given layout.
int Size(Layout layout);

// Returns Axis for the given index and fixed layout.
template <Layout T>
constexpr Axis GetAxis(int index);

// Returns axis for the given layout and index.
Axis GetAxis(Layout layout, int32_t index);

// Returns axis index for the given axis and fixed layout.
template <Layout T>
constexpr int GetAxisIndex(Axis axis);

// Returns axis index for the given layout and axis.
int GetAxisIndex(Layout layout, Axis axis);

// Checks if fixed layout has given axis
template <Layout T>
constexpr bool HasAxis(Axis axis);

// Checks if given layout has given axis
bool HasAxis(Layout layout, Axis axis);

// Stores Layout(axis set and order) and value for dimensions.
struct Shape {
  Shape() : layout(Layout::UNKNOWN), dimensions() {}

  explicit Shape(Layout t) : layout(t), dimensions(Size(t)) {}

  Shape(Layout t, std::vector<int32_t> d)
      : layout(t), dimensions(std::move(d)) {}

  bool operator==(const Shape& other) const {
    return (layout == other.layout) && (dimensions == other.dimensions);
  }

  bool operator!=(const Shape& other) const { return !operator==(other); }

  // All methods below are matching same methods defined in StrongShape to
  // make sure generic algorithms work both ways.

  // Returns back a dimension or -1 if it is not found.
  template <Axis D>
  int32_t get() const;
  int32_t get(Axis axis) const;

  template <Axis D>
  bool set(int32_t t);
  bool set(Axis axis, int32_t t);

  Axis axis(int index) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_0(mht_0_v, 291, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "axis");
 return GetAxis(layout, index); }

  int index(Axis axis) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_1(mht_1_v, 296, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "index");
 return GetAxisIndex(layout, axis); }

  bool has(Axis axis) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_2(mht_2_v, 301, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "has");
 return HasAxis(layout, axis); }

  int64_t DimensionsProduct() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_3(mht_3_v, 306, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "DimensionsProduct");

    return std::accumulate(dimensions.begin(), dimensions.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  Layout layout = Layout::UNKNOWN;

  std::vector<int32_t> dimensions;
};

std::string ToString(const Shape& s);

// StrongShape provides convenient explicit access to dimensions stored in
// shape, e.g. StrongShape<Layout::HW> s; provides s.h and s.w accessors.
//
// There is a conversion possible both ways between Shape and StrongShape.
//
//   OIHW oihw;  // specific shape
//   Shape l = oihw.ToShape();
//
//   OHWI other;  // notice not the same but compatible shape.
//   if (!other.Adopt(l)) {
//     // error handling
//   }
//
// StrongShape supports the following set of operations:
//
//   // Returns number of axis in the shape class.
//   static constexpr int size();
//
//   // Returns Axis for the given index or Axis::UNKNOWN if index
//   // falls outside of the defined range in this shape.
//   static constexpr Axis axis(int index);
//
//   // Returns index for the given axis or -1 if axis is not defined in this
//   // shape.
//   static constexpr int index(Axis axis);
//
//   // Getters
//   int32_t get(int index) const;
//   int32_t get(Axis axis) const;
//   int32_t get<Axis>() const;
//
//   // Setters that return false if set was not successful.
//   bool set(int index, int32_t v);
//   bool set(Axis axis, int32_t v);
//   bool set<Axis>(int32_t v);
//
//   // Returns shape's layout.
//   static const Layout layout;
//
//   // Turns specific shape into generic shape.
//   Shape ToShape() const;
//
//   // Copies all dimensions from the given shape.
//   bool Adopt(const Shape&);
//
template <Layout L>
struct StrongShape;

using Scalar = StrongShape<Layout::SCALAR>;
using Linear = StrongShape<Layout::LINEAR>;
using HW = StrongShape<Layout::HW>;
using HWD = StrongShape<Layout::HWD>;

// Common tensor shape for CNN models working with images.
using CHW = StrongShape<Layout::CHW>;
using HWC = StrongShape<Layout::HWC>;
using HWDC = StrongShape<Layout::HWDC>;
using BHWC = StrongShape<Layout::BHWC>;
using BHWDC = StrongShape<Layout::BHWDC>;

// Tensor shape used in convolution_2d weights.
using OIHW = StrongShape<Layout::OIHW>;
using OHWI = StrongShape<Layout::OHWI>;
using IHWO = StrongShape<Layout::IHWO>;
using IOHW = StrongShape<Layout::IOHW>;

// Tensor shape used in convolution_3d weights.
using OHWDI = StrongShape<Layout::OHWDI>;

// -----------------------------------------------------------------------------
// Everything below are internal implementation details.
// -----------------------------------------------------------------------------

namespace internal_shape {

template <Axis T>
struct AxisTraits;

#define TFLITE_GPU_AXIS_TRAITS(AxisName, HolderName)    \
  template <>                                           \
  struct AxisTraits<Axis::AxisName> {                   \
    struct Holder {                                     \
      int32_t HolderName;                               \
                                                        \
     protected:                                         \
      int32_t operator()() const { return HolderName; } \
      void operator()(int32_t v) { HolderName = v; }    \
    };                                                  \
                                                        \
    using dimension_holder_type = Holder;               \
  }

TFLITE_GPU_AXIS_TRAITS(CHANNELS, c);
TFLITE_GPU_AXIS_TRAITS(HEIGHT, h);
TFLITE_GPU_AXIS_TRAITS(WIDTH, w);
TFLITE_GPU_AXIS_TRAITS(INPUT_CHANNELS, i);
TFLITE_GPU_AXIS_TRAITS(OUTPUT_CHANNELS, o);
TFLITE_GPU_AXIS_TRAITS(BATCH, b);
TFLITE_GPU_AXIS_TRAITS(VALUE, v);
TFLITE_GPU_AXIS_TRAITS(DEPTH, d);

#undef TFLITE_GPU_AXIS_TRAITS

template <int N, Axis... As>
struct StrongShapeImpl;

template <int N>
struct StrongShapeImpl<N> {
  static constexpr int size() { return N; }

  static constexpr Axis axis(int) { return Axis::UNKNOWN; }

  static constexpr int index(Axis) { return -1; }

  static constexpr bool has(Axis) { return false; }

  int32_t get(Axis) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_4(mht_4_v, 437, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");
 return -1; }

  int32_t get(int) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_5(mht_5_v, 442, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");
 return -1; }

  template <Axis B>
  int32_t get() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_6(mht_6_v, 448, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");

    return -1;
  }

  bool set(Axis, int32_t) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_7(mht_7_v, 455, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");
 return false; }

  bool set(int, int32_t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_8(mht_8_v, 460, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");
 return false; }

  template <Axis B>
  bool set(int32_t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_9(mht_9_v, 466, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");

    return false;
  }
};

// Used to deduce number of axis, and to be a child of a proper holder to
// provide access to the dimension by name
template <int N, Axis A, Axis... As>
struct StrongShapeImpl<N, A, As...>
    : public AxisTraits<A>::dimension_holder_type,
      public StrongShapeImpl<N + 1, As...> {
  using dimension_holder_type = typename AxisTraits<A>::dimension_holder_type;

  using rest_type = StrongShapeImpl<N + 1, As...>;

  StrongShapeImpl() : dimension_holder_type{0}, rest_type() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_10(mht_10_v, 484, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "StrongShapeImpl");
}

  template <typename... Ts>
  explicit StrongShapeImpl(int32_t t, Ts... ts)
      : dimension_holder_type{t}, rest_type(ts...) {}

  static constexpr Axis axis(int index) {
    return index == N ? A : rest_type::axis(index);
  }

  static constexpr int index(Axis axis) {
    return axis == A ? N : rest_type::index(axis);
  }

  static constexpr bool has(Axis axis) {
    return axis == A ? true : rest_type::has(axis);
  }

  int32_t get(Axis axis) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_11(mht_11_v, 505, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");

    return axis == A ? dimension_holder_type::operator()()
                     : rest_type::get(axis);
  }

  template <Axis B>
  int32_t get() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_12(mht_12_v, 514, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");

    return B == A ? dimension_holder_type::operator()()
                  : rest_type::template get<B>();
  }

  int32_t get(int index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_13(mht_13_v, 522, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "get");

    return index == N ? dimension_holder_type::operator()()
                      : rest_type::get(index);
  }

  bool set(Axis axis, int32_t t) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_14(mht_14_v, 530, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");

    if (axis == A) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::set(axis, t);
  }

  bool set(int index, int32_t t) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_15(mht_15_v, 541, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");

    if (index == N) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::set(index, t);
  }

  template <Axis B>
  bool set(int32_t t) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_16(mht_16_v, 553, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "set");

    if (A == B) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::template set<B>(t);
  }
};

template <Layout T>
struct LayoutTraits;

#define TFLITE_GPU_LAYOUT_TRAITS(LayoutName, ...)              \
  template <>                                                  \
  struct LayoutTraits<Layout::LayoutName> {                    \
    using strong_shape_type = StrongShapeImpl<0, __VA_ARGS__>; \
  }

TFLITE_GPU_LAYOUT_TRAITS(HW, Axis::HEIGHT, Axis::WIDTH);
TFLITE_GPU_LAYOUT_TRAITS(HWD, Axis::HEIGHT, Axis::WIDTH, Axis::DEPTH);
TFLITE_GPU_LAYOUT_TRAITS(OHWI, Axis::OUTPUT_CHANNELS, Axis::HEIGHT, Axis::WIDTH,
                         Axis::INPUT_CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(OIHW, Axis::OUTPUT_CHANNELS, Axis::INPUT_CHANNELS,
                         Axis::HEIGHT, Axis::WIDTH);
TFLITE_GPU_LAYOUT_TRAITS(IOHW, Axis::INPUT_CHANNELS, Axis::OUTPUT_CHANNELS,
                         Axis::HEIGHT, Axis::WIDTH);
TFLITE_GPU_LAYOUT_TRAITS(IHWO, Axis::INPUT_CHANNELS, Axis::HEIGHT, Axis::WIDTH,
                         Axis::OUTPUT_CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(CHW, Axis::CHANNELS, Axis::HEIGHT, Axis::WIDTH);
TFLITE_GPU_LAYOUT_TRAITS(HWC, Axis::HEIGHT, Axis::WIDTH, Axis::CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(HWDC, Axis::HEIGHT, Axis::WIDTH, Axis::DEPTH,
                         Axis::CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(LINEAR, Axis::VALUE);
TFLITE_GPU_LAYOUT_TRAITS(SCALAR, Axis::VALUE);
TFLITE_GPU_LAYOUT_TRAITS(BHWC, Axis::BATCH, Axis::HEIGHT, Axis::WIDTH,
                         Axis::CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(BHWDC, Axis::BATCH, Axis::HEIGHT, Axis::WIDTH,
                         Axis::DEPTH, Axis::CHANNELS);
TFLITE_GPU_LAYOUT_TRAITS(OHWDI, Axis::OUTPUT_CHANNELS, Axis::HEIGHT,
                         Axis::WIDTH, Axis::DEPTH, Axis::INPUT_CHANNELS);

#undef TFLITE_GPU_LAYOUT_TRAITS

template <>
struct LayoutTraits<Layout::UNKNOWN> {
  using strong_shape_type = StrongShapeImpl<0>;
};

template <Axis A>
struct DimensionGetterFixedAxisFunc {
  template <Layout T>
  int32_t operator()() const {
    constexpr int i = GetAxisIndex<T>(A);
    return i >= 0 && i < l->dimensions.size() ? l->dimensions[i] : -1;
  }
  const Shape* l;
};

struct DimensionGetterFunc {
  template <Layout T>
  int32_t operator()() const {
    int i = GetAxisIndex<T>(axis);
    return i >= 0 && i < l->dimensions.size() ? l->dimensions[i] : -1;
  }
  Axis axis;
  const Shape* l;
};

template <Axis A>
struct DimensionSetterFixedAxisFunc {
  template <Layout T>
  bool operator()() const {
    constexpr int i = GetAxisIndex<T>(A);
    if (i >= 0 && i < l->dimensions.size()) {
      l->dimensions[i] = v;
      return true;
    }
    return false;
  }
  Shape* l;
  int32_t v;
};

struct DimensionSetterFunc {
  template <Layout T>
  bool operator()() const {
    int i = GetAxisIndex<T>(axis);
    if (i >= 0 && i < l->dimensions.size()) {
      l->dimensions[i] = v;
      return true;
    }
    return false;
  }
  Axis axis;
  Shape* l;
  int32_t v;
};

template <Layout L>
struct ToShapeFunc {
  template <Layout T>
  bool operator()() const {
    for (int i = 0; i < StrongShape<L>::size(); ++i) {
      int index = GetAxisIndex<T>(StrongShape<L>::axis(i));
      if (index < 0) return false;
      shape->set(i, l.dimensions[index]);
    }
    return true;
  }

  StrongShape<L>* shape;
  const Shape& l;
};

}  // namespace internal_shape

// template <Axis... As>
template <Layout L>
struct StrongShape : public internal_shape::LayoutTraits<L>::strong_shape_type {
  using strong_shape_type =
      typename internal_shape::LayoutTraits<L>::strong_shape_type;
  StrongShape() = default;

  template <typename... Ts>
  explicit StrongShape(Ts... t) : strong_shape_type(t...) {}

  constexpr static Layout layout = L;

  bool operator==(const StrongShape<L>& shape) const {
    // TODO(akulik): implement better alternative.
    return this->ToShape() == shape.ToShape();
  }

  bool operator!=(const StrongShape<L>& shape) const {
    // TODO(akulik): implement better alternative.
    return this->ToShape() != shape.ToShape();
  }
  bool empty() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_17(mht_17_v, 693, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "empty");
 return DimensionsProduct() == 0; }

  // Turns StrongShape into generic shape.
  Shape ToShape() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_18(mht_18_v, 699, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "ToShape");

    std::vector<int32_t> dimensions(StrongShape::size());
    for (int i = 0; i < StrongShape::size(); ++i) {
      dimensions[i] = StrongShape::get(i);
    }
    return Shape(L, std::move(dimensions));
  }

  // @return all dimensions multiplied
  int64_t DimensionsProduct() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_19(mht_19_v, 711, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "DimensionsProduct");

    int64_t product = 1;
    for (int i = 0; i < StrongShape::size(); ++i) {
      product *= StrongShape::get(i);
    }
    return product;
  }

  // Translates given coordinates of the layout into a linear index assuming
  // dimensions are sorted in tensor access order e.g. if you access
  // foobar[i][j][k] order of coordinates should be i,j,k.
  int64_t LinearIndex(
      const std::array<int32_t, StrongShape::size()>& coordinates) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_20(mht_20_v, 726, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "LinearIndex");

    int64_t index = coordinates[0];
    for (int i = 1; i < StrongShape::size(); ++i) {
      index = index * StrongShape::get(i) + coordinates[i];
    }
    return index;
  }

  // Copies all dimensions from the given generic shape into specific shape.
  // It requires shape to have all axis defined in the given
  // StrongShape. For example:
  //   - If this shape is OHWI but given shape is OIHW, Adopt will copy all
  //     dimensions and return true.
  //   - If this shape is OIHW but input shape is HW, Adopt will copy H and W
  //     dimensions and return true, but if this shape is HW and given shape
  //     OIHW, then Adopt will return false because not all axis are present in
  //     the input shape.
  //
  // @return false if generic shape is not compatible.
  bool Adopt(const Shape& shape) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_21(mht_21_v, 748, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "Adopt");

    return DispatchByLayout(shape.layout,
                            internal_shape::ToShapeFunc<L>{this, shape});
  }

  // For all axis defined in a given shape copies values to this shape.
  // Therefore, it is possible to copy dimensions from CHW to BCHW, but not
  // the other way around.
  //
  // BCHW bchw;
  // CHW chw;
  // bchw.CopyAllGivenAxis(chw);  --> true
  // chw.CopyAllGivenAxis(bchw);  --> false
  //
  // @return false if axis in source shape is not defined here, thus value
  //         was not copied.
  template <Layout B>
  bool CopyAllGivenAxis(const StrongShape<B>& source) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_22(mht_22_v, 768, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "CopyAllGivenAxis");

    for (int i = 0; i < source.size(); ++i) {
      if (!StrongShape::set(source.axis(i), source.get(i))) {
        return false;
      }
    }
    return true;
  }

  // For all axis defined in this shape copies values from the given shape.
  //
  // BCHW bchw;
  // CHW chw;
  // bchw.CopyAllDefinedAxis(chw);  --> false
  // chw.CopyAllDefinedAxis(bchw);  --> true
  //
  // @return false if given shape does not have axis defined here,
  //         therefore a value was not copied.
  template <Layout B>
  bool CopyAllDefinedAxis(const StrongShape<B>& source) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_23(mht_23_v, 790, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "CopyAllDefinedAxis");

    for (int i = 0; i < StrongShape::size(); ++i) {
      int source_index = source.index(StrongShape::axis(i));
      if (source_index < 0) {
        return false;
      }
      StrongShape::set(i, source.get(source_index));  // always true
    }
    return true;
  }

  // Copies values only for matching axis.
  template <Layout B>
  void CopyMatchingAxis(const StrongShape<B>& source) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_24(mht_24_v, 806, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "CopyMatchingAxis");

    for (int i = 0; i < StrongShape::size(); ++i) {
      StrongShape::set(source.axis(i), source.get(i));
    }
  }

  // AbslHash function for using in flat hash containers.
  template <typename H>
  friend H AbslHashValue(H hash_state, const StrongShape& strong_shape) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_25(mht_25_v, 817, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "AbslHashValue");

    for (size_t i = 0; i < strong_shape.size(); ++i) {
      hash_state = H::combine(std::move(hash_state), strong_shape.get(i));
    }
    return hash_state;
  }
};

template <Layout T>
inline std::string ToString(const StrongShape<T>& s) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_26(mht_26_v, 829, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "ToString");

  return ToString(s.ToShape());
}

template <Layout L>
constexpr Layout StrongShape<L>::layout;

template <class F>
auto DispatchByLayout(Layout type, F f)
    -> decltype(f.template operator()<Layout::UNKNOWN>()) {
  switch (type) {
    case Layout::HW:
      return f.template operator()<Layout::HW>();
    case Layout::HWD:
      return f.template operator()<Layout::HWD>();
    case Layout::HWC:
      return f.template operator()<Layout::HWC>();
    case Layout::HWDC:
      return f.template operator()<Layout::HWDC>();
    case Layout::CHW:
      return f.template operator()<Layout::CHW>();
    case Layout::OIHW:
      return f.template operator()<Layout::OIHW>();
    case Layout::IOHW:
      return f.template operator()<Layout::IOHW>();
    case Layout::OHWI:
      return f.template operator()<Layout::OHWI>();
    case Layout::IHWO:
      return f.template operator()<Layout::IHWO>();
    case Layout::LINEAR:
      return f.template operator()<Layout::LINEAR>();
    case Layout::SCALAR:
      return f.template operator()<Layout::SCALAR>();
    case Layout::BHWC:
      return f.template operator()<Layout::BHWC>();
    case Layout::BHWDC:
      return f.template operator()<Layout::BHWDC>();
    case Layout::OHWDI:
      return f.template operator()<Layout::OHWDI>();
    case Layout::UNKNOWN:
      return f.template operator()<Layout::UNKNOWN>();
  }
}

template <Layout T>
constexpr int Size() {
  return StrongShape<T>::size();
}

template <Layout T>
constexpr Axis GetAxis(int index) {
  return StrongShape<T>::axis(index);
}

template <Layout T>
constexpr int GetAxisIndex(Axis axis) {
  return StrongShape<T>::index(axis);
}

template <Layout T>
constexpr bool HasAxis(Axis axis) {
  return StrongShape<T>::has(axis);
}

template <Axis D>
inline int32_t Shape::get() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_27(mht_27_v, 897, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "Shape::get");

  return DispatchByLayout(
      layout, internal_shape::DimensionGetterFixedAxisFunc<D>{this});
}

inline int32_t Shape::get(Axis axis) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_28(mht_28_v, 905, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "Shape::get");

  return DispatchByLayout(layout,
                          internal_shape::DimensionGetterFunc{axis, this});
}

template <Axis D>
inline bool Shape::set(int32_t t) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_29(mht_29_v, 914, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "Shape::set");

  return DispatchByLayout(
      layout, internal_shape::DimensionSetterFixedAxisFunc<D>{this, t});
}

inline bool Shape::set(Axis axis, int32_t t) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_30(mht_30_v, 922, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "Shape::set");

  return DispatchByLayout(layout,
                          internal_shape::DimensionSetterFunc{axis, this, t});
}

template <Layout T>
std::ostream& operator<<(std::ostream& ostream, const StrongShape<T>& shape) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTh mht_31(mht_31_v, 931, "", "./tensorflow/lite/delegates/gpu/common/shape.h", "operator<<");

  ostream << ToString(shape);
  return ostream;
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SHAPE_H_
