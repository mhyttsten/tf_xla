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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh() {
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


#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace array_impl {

// conjunction
//
// Performs a compile-time logical AND operation on the passed types (which
// must have  `::value` members convertible to `bool`. Short-circuits if it
// encounters any `false` members (and does not compare the `::value` members
// of any remaining arguments).
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::conjunction` metafunction.
template <typename... Ts>
struct conjunction;

template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};

template <>
struct conjunction<> : std::true_type {};

// A type trait that is valid when all elements in a parameter pack are of
// integral type. Not using an alias template to work around MSVC 14.00 bug.
template <typename... Ts>
struct pack_is_integral : conjunction<std::is_integral<Ts>...> {};

// Compares three same-sized vectors elementwise. For each item in `values`,
// returns false if any of values[i] is outside the half-open range [starts[i],
// ends[i]).
template <typename C1, typename C2, typename C3>
bool all_inside_range(const C1& values, const C2& range_starts,
                      const C3& range_ends) {
  for (size_t i = 0, e = values.size(); i < e; ++i) {
    if (values[i] < range_starts[i] || values[i] >= range_ends[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace array_impl

// General N dimensional array class with arbitrary value type.
template <typename T>
class Array {
 public:
  // Type inference can have a hard time parsing very deep initializer list
  // nests, especially if one or more dimensions is one as the compiler just
  // sees a single-element integer initializer. These typedefs allow casting
  // explicitly with less typing.
  using InitializerList1D = std::initializer_list<T>;
  using InitializerList2D = std::initializer_list<InitializerList1D>;
  using InitializerList3D = std::initializer_list<InitializerList2D>;
  using InitializerList4D = std::initializer_list<InitializerList3D>;

  using value_type = T;

  // Creates a new array with the specified dimensions and initialized elements.
  explicit Array(absl::Span<const int64_t> sizes)
      : sizes_(sizes.begin(), sizes.end()), values_(new T[num_elements()]()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_0(mht_0_v, 268, "", "./tensorflow/compiler/xla/array.h", "Array");
}

  // Creates a new array with the specified dimensions and specified value for
  // every cell.
  Array(absl::Span<const int64_t> sizes, T value)
      : sizes_(sizes.begin(), sizes.end()), values_(new T[num_elements()]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_1(mht_1_v, 276, "", "./tensorflow/compiler/xla/array.h", "Array");

    Fill(value);
  }

  // Creates a 2D array from the given nested initializer list. The outer
  // initializer list is the first dimension, the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array(InitializerList2D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size()})) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_2(mht_2_v, 287, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        values_[idx] = it2;
        ++idx;
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 1D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<T2> values)
      : Array(ToInt64Vector({values.size()})) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_3(mht_3_v, 310, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      values_[idx] = static_cast<T>(it1);
      ++idx;
    }
    CHECK(idx == num_elements());
  }

  // Creates a 2D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<std::initializer_list<T2>> values)
      : Array(ToInt64Vector({values.size(), values.begin()->size()})) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_4(mht_4_v, 331, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        values_[idx] = static_cast<T>(it2);
        ++idx;
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 3D array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  Array(InitializerList3D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size()})) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_5(mht_5_v, 349, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          values_[idx] = it3;
          ++idx;
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 3D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
            values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size()})) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_6(mht_6_v, 376, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          values_[idx] = static_cast<T>(it3);
          ++idx;
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 4D array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  Array(InitializerList4D values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size(),
                             values.begin()->begin()->begin()->size()})) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_7(mht_7_v, 397, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          for (const auto& it4 : it3) {
            values_[idx] = it4;
            ++idx;
          }
        }
      }
    }
    CHECK(idx == num_elements());
  }

  // Creates a 4D array of a floating-point type (half, bfloat16, float,
  // or double) from an initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array(std::initializer_list<
        std::initializer_list<std::initializer_list<std::initializer_list<T2>>>>
            values)
      : Array(ToInt64Vector({values.size(), values.begin()->size(),
                             values.begin()->begin()->size(),
                             values.begin()->begin()->begin()->size()})) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_8(mht_8_v, 428, "", "./tensorflow/compiler/xla/array.h", "Array");

    int64_t idx = 0;
    for (const auto& it1 : values) {
      for (const auto& it2 : it1) {
        for (const auto& it3 : it2) {
          for (const auto& it4 : it3) {
            values_[idx] = static_cast<T>(it4);
            ++idx;
          }
        }
      }
    }
    CHECK(idx == num_elements());
  }

  Array(const Array<T>& other)
      : sizes_(other.sizes_), values_(new T[num_elements()]) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_9(mht_9_v, 447, "", "./tensorflow/compiler/xla/array.h", "Array");

    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
  }

  Array<T>& operator=(const Array<T>& other) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_10(mht_10_v, 455, "", "./tensorflow/compiler/xla/array.h", "=");

    sizes_ = other.sizes_;
    values_.reset(new T[num_elements()]);
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
    return *this;
  }

  // Fills the array with the specified value.
  void Fill(const T& value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_11(mht_11_v, 467, "", "./tensorflow/compiler/xla/array.h", "Fill");

    std::fill(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with sequentially increasing values.
  void FillIota(const T& value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_12(mht_12_v, 475, "", "./tensorflow/compiler/xla/array.h", "FillIota");

    std::iota(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with a repeating sequence:
  //   [value, value + 1, ..., value + length - 1, value, ... ]
  void FillRepeatedIota(const T& value, int64_t length) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_13(mht_13_v, 484, "", "./tensorflow/compiler/xla/array.h", "FillRepeatedIota");

    for (int64_t i = 0; i < num_elements(); i += length) {
      std::iota(&values_[i], &values_[std::min(i + length, num_elements())],
                value);
    }
  }

  // Fills the array with the sequence i*multiplier for i=0,1,...
  void FillWithMultiples(const T& multiplier) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_14(mht_14_v, 495, "", "./tensorflow/compiler/xla/array.h", "FillWithMultiples");

    for (int64_t i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(i) * multiplier;
    }
  }

  // Fills the array with random normal variables with the specified mean.
  void FillRandom(const T& stddev, double mean = 0.0, int seed = 12345) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_15(mht_15_v, 505, "", "./tensorflow/compiler/xla/array.h", "FillRandom");

    FillRandomDouble(static_cast<double>(stddev), mean, seed);
  }

  void FillRandomDouble(double stddev, double mean = 0.0, int seed = 12345) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_16(mht_16_v, 512, "", "./tensorflow/compiler/xla/array.h", "FillRandomDouble");

    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean, stddev);
    for (int64_t i = 0; i < num_elements(); ++i) {
      if (std::is_same<T, bool>()) {
        values_[i] = static_cast<T>(distribution(g) > 0.0);
      } else {
        values_[i] = static_cast<T>(distribution(g));
      }
    }
  }

  // Fills the array with random uniform variables in the [min_value, max_value]
  // range. Defined for integral types.
  template <typename = typename std::enable_if<std::is_integral<T>::value>>
  void FillRandomUniform(const T& min_value, const T& max_value,
                         int seed = 12345) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_17(mht_17_v, 531, "", "./tensorflow/compiler/xla/array.h", "FillRandomUniform");

    std::mt19937 g(seed);
    std::uniform_int_distribution<T> distribution(min_value, max_value);
    for (int64_t i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(distribution(g));
    }
  }

  // Sets all the values in the array to values specified in the container.
  template <typename Container = std::initializer_list<T>>
  void SetValues(const Container& container) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_18(mht_18_v, 544, "", "./tensorflow/compiler/xla/array.h", "SetValues");

    CHECK_EQ(std::distance(std::begin(container), std::end(container)),
             num_elements());
    std::copy(std::begin(container), std::end(container), &values_[0]);
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array.
  void Each(std::function<void(absl::Span<const int64_t>, T*)> f) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_19(mht_19_v, 555, "", "./tensorflow/compiler/xla/array.h", "Each");

    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, &values_[i]);
    }
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  void Each(std::function<void(absl::Span<const int64_t>, T)> f) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_20(mht_20_v, 566, "", "./tensorflow/compiler/xla/array.h", "Each");

    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      f(index, values_[i]);
    }
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the
  // array. If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(std::function<Status(absl::Span<const int64_t>, T*)> f) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_21(mht_21_v, 579, "", "./tensorflow/compiler/xla/array.h", "EachStatus");

    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, &values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return Status::OK();
  }

  // Invokes a callback with the (indices, value) for each cell in the array.
  // If a callback returns a non-OK status, returns that else returns
  // Status::OK().
  Status EachStatus(
      std::function<Status(absl::Span<const int64_t>, T)> f) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_22(mht_22_v, 597, "", "./tensorflow/compiler/xla/array.h", "EachStatus");

    std::vector<int64_t> index(sizes_.size());
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      Status s = f(index, values_[i]);
      if (!s.ok()) {
        return s;
      }
    }
    return Status::OK();
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  //
  // The type trait is required to avoid this overload participating too
  // eagerly; a parameter pack can take zero or more elements, so we must
  // restrict this to only parameter packs that are all of integral type.
  template <typename... Dims>
  typename std::enable_if<array_impl::pack_is_integral<Dims...>::value,
                          const T&>::type
  operator()(Dims... dims) const {
    // We are using a std::array to avoid having to allocate memory in this
    // function for performance reasons.
    std::array<int64_t, sizeof...(dims)> indexes{
        {static_cast<int64_t>(dims)...}};
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  template <typename... Dims>
  typename std::enable_if<array_impl::pack_is_integral<Dims...>::value,
                          T&>::type
  operator()(Dims... dims) {
    // We are using a std::array to avoid having to allocate memory in this
    // function for performance reasons.
    std::array<int64_t, sizeof...(dims)> indexes{
        {static_cast<int64_t>(dims)...}};
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  const T& operator()(absl::Span<const int64_t> indexes) const {
    return values_[calculate_index(indexes)];
  }

  // Returns the value at the cell specified by the indexes. The number of
  // arguments have to match with the number of dimensions for the array.
  T& operator()(absl::Span<const int64_t> indexes) {
    return values_[calculate_index(indexes)];
  }

  // Low-level accessor for stuff like memcmp, handle with care. Returns pointer
  // to the underlying storage of the array (similarly to std::vector::data()).
  T* data() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_23(mht_23_v, 655, "", "./tensorflow/compiler/xla/array.h", "data");

    // TODO(tberghammer): Get rid of the const_cast. Currently it is needed
    // because the Eigen backend needs a non-const pointers even for reading
    // from the array.
    return const_cast<Array*>(this)->values_.get();
  }

  // Returns the size of the dimension at the given index.
  int64_t dim(int64_t n) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_24(mht_24_v, 666, "", "./tensorflow/compiler/xla/array.h", "dim");

    const int64_t sizes_size = sizes_.size();
    CHECK(n < sizes_size);
    return sizes_[n];
  }

  // Returns a vector containing the dimensions of the array.
  const std::vector<int64_t>& dimensions() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_25(mht_25_v, 676, "", "./tensorflow/compiler/xla/array.h", "dimensions");
 return sizes_; }

  int64_t num_dimensions() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_26(mht_26_v, 681, "", "./tensorflow/compiler/xla/array.h", "num_dimensions");
 return sizes_.size(); }

  // Returns the total number of elements in the array.
  int64_t num_elements() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_27(mht_27_v, 687, "", "./tensorflow/compiler/xla/array.h", "num_elements");

    return std::accumulate(sizes_.begin(), sizes_.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  const T* begin() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_28(mht_28_v, 695, "", "./tensorflow/compiler/xla/array.h", "begin");
 return &values_[0]; }
  T* begin() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_29(mht_29_v, 699, "", "./tensorflow/compiler/xla/array.h", "begin");
 return &values_[0]; }
  const T* end() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_30(mht_30_v, 703, "", "./tensorflow/compiler/xla/array.h", "end");
 return &values_[num_elements()]; }
  T* end() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_31(mht_31_v, 707, "", "./tensorflow/compiler/xla/array.h", "end");
 return &values_[num_elements()]; }

  bool operator==(const Array<T>& other) const {
    if (sizes_.size() != other.sizes_.size()) {
      return false;
    }
    for (int64_t i = 0, end = sizes_.size(); i < end; ++i) {
      if (sizes_[i] != other.sizes_[i]) {
        return false;
      }
    }
    for (int64_t i = 0; i < num_elements(); ++i) {
      if (values_[i] != other.values_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Array<T>& other) const { return !(*this == other); }

  // Performs the equivalent of a slice operation on this array.
  Array<T> Slice(absl::Span<const int64_t> starts,
                 absl::Span<const int64_t> limits) const {
    CHECK_EQ(starts.size(), num_dimensions());
    CHECK_EQ(limits.size(), num_dimensions());

    std::vector<int64_t> sizes;
    std::transform(starts.begin(), starts.end(), limits.begin(),
                   std::back_inserter(sizes),
                   [](int64_t start, int64_t limit) { return limit - start; });
    Array<T> result(sizes);

    std::vector<int64_t> index(sizes_.size());
    int64_t slice_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index, starts, limits)) {
        // Even though the bounds of result are different to our bounds, we're
        // iterating in the same order. So we can simply write successive linear
        // indices instead of recalculating a multi-dimensional index.
        result.values_[slice_i++] = values_[i];
      }
    }
    return result;
  }

  // Performs the equivalent of a DynamicUpdateSlice in-place on this array.
  void UpdateSlice(const Array<T>& from,
                   absl::Span<const int64_t> start_indices) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_32(mht_32_v, 758, "", "./tensorflow/compiler/xla/array.h", "UpdateSlice");

    CHECK_EQ(from.num_dimensions(), num_dimensions());
    std::vector<int64_t> limit_indices;
    std::transform(start_indices.begin(), start_indices.end(),
                   from.dimensions().begin(), std::back_inserter(limit_indices),
                   std::plus<int64_t>{});
    std::vector<int64_t> index(sizes_.size());
    int64_t from_i = 0;
    for (int64_t i = 0; i < num_elements(); ++i, next_index(&index)) {
      if (array_impl::all_inside_range(index, start_indices, limit_indices)) {
        // Even though the bounds of from are different to our bounds, we're
        // iterating in the same order. So we can simply write successive linear
        // indices instead of recalculating a multi-dimensional index.
        values_[i] = from.values_[from_i++];
      }
    }
  }

  // Performs an in-place reshape, modifying the dimensions but not the
  // underlying data.
  void Reshape(absl::Span<const int64_t> new_dimensions) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_33(mht_33_v, 781, "", "./tensorflow/compiler/xla/array.h", "Reshape");

    int64_t old_num_elements = num_elements();
    sizes_ = std::vector<int64_t>(new_dimensions.begin(), new_dimensions.end());
    CHECK_EQ(num_elements(), old_num_elements);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Array& array) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_34(mht_34_v, 791, "", "./tensorflow/compiler/xla/array.h", "AbslHashValue");

    return H::combine(std::move(h), absl::MakeSpan(array.begin(), array.end()),
                      array.dimensions());
  }

  // Returns a string representation of the array suitable for debugging.
  std::string ToString() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_35(mht_35_v, 800, "", "./tensorflow/compiler/xla/array.h", "ToString");

    if (sizes_.empty()) {
      return "";
    }
    std::vector<std::string> pieces;
    std::vector<int64_t> index(sizes_.size());
    do {
      // Emit leading spaces and opening square brackets
      if (index.back() == 0) {
        for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
          if (i == 0 || index[i - 1] != 0) {
            for (int64_t j = 0; j < sizes_.size(); ++j) {
              pieces.push_back(j < i ? " " : "[");
            }
            break;
          }
        }
      }
      int value_index = calculate_index(index);
      if (value_index < num_elements()) {
        pieces.push_back(absl::StrCat(values_[value_index]));
      }

      // Emit comma if it isn't the last element
      if (index.back() < sizes_.back() - 1) {
        pieces.push_back(", ");
      }

      // Emit closing square brackets
      for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
        if (index[i] < sizes_[i] - 1) {
          break;
        }
        pieces.push_back("]");
        if (i != 0 && index[i - 1] < sizes_[i - 1] - 1) {
          pieces.push_back(",\n");
        }
      }
    } while (next_index(&index));
    return absl::StrJoin(pieces, "");
  }

 private:
  // Converts an initializer_list of type U to a vector of type int64_t. Used by
  // the initializer list based constructors to convert the size type into
  // int64_t to be passed to the size based constructor.
  template <typename U>
  static std::vector<int64_t> ToInt64Vector(
      const std::initializer_list<U>& data) {
    return std::vector<int64_t>(data.begin(), data.end());
  }

  // Returns the linear index from the list of per-dimension indexes. Function
  // is templated so can be used with an std::array from operator() to avoid
  // memory allocation.
  // The returned value may be larger than or equal to the number of elements if
  // the indexes exceed the array's corresponding dimension size.
  template <typename U>
  int64_t calculate_index(const U& indexes) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_36(mht_36_v, 861, "", "./tensorflow/compiler/xla/array.h", "calculate_index");

    CHECK_EQ(sizes_.size(), indexes.size());
    int64_t index = 0;
    for (int64_t i = 0; i < sizes_.size(); ++i) {
      index *= sizes_[i];
      index += indexes[i];
    }
    return index;
  }

  // Advances the specified set of indexes and returns true if we haven't
  // wrapped around (i.e. result isn't {0, 0, ...}).
  bool next_index(std::vector<int64_t>* index) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarrayDTh mht_37(mht_37_v, 876, "", "./tensorflow/compiler/xla/array.h", "next_index");

    CHECK_EQ(index->size(), sizes_.size());
    for (int64_t i = sizes_.size() - 1; i >= 0; --i) {
      (*index)[i]++;
      if ((*index)[i] < sizes_[i]) {
        return true;
      }
      (*index)[i] = 0;
    }
    return false;
  }

  std::vector<int64_t> sizes_;
  std::unique_ptr<T[]> values_;
};

// Specialization of FillRandom() method for complex64 type. Uses real part of
// the stddev parameter as the standard deviation value.
template <>
void Array<complex64>::FillRandom(const complex64& stddev, const double mean,
                                  const int seed);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY_H_
