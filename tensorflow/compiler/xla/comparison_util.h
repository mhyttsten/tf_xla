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

#ifndef TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh() {
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


#include "absl/base/macros.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class Comparison {
 public:
  // Represents type of comparison
  enum class Type : uint8_t {
    kFloat,
    kFloatTotalOrder,
    kSigned,
    kUnsigned,
  };

  // Represents different comparison operations.
  enum class Direction : uint8_t {
    kEq,
    kNe,
    kGe,
    kGt,
    kLe,
    kLt,
  };

  Comparison() = delete;
  explicit Comparison(Direction dir, Type type) : dir_(dir), type_(type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/comparison_util.h", "Comparison");
}
  explicit Comparison(Direction dir, PrimitiveType type);

  Direction GetDirection() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/comparison_util.h", "GetDirection");
 return dir_; }
  Type GetType() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/comparison_util.h", "GetType");
 return type_; }

  inline bool IsEq() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_3(mht_3_v, 232, "", "./tensorflow/compiler/xla/comparison_util.h", "IsEq");
 return dir_ == Direction::kEq; }
  inline bool IsNe() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_4(mht_4_v, 236, "", "./tensorflow/compiler/xla/comparison_util.h", "IsNe");
 return dir_ == Direction::kNe; }
  inline bool IsGe() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_5(mht_5_v, 240, "", "./tensorflow/compiler/xla/comparison_util.h", "IsGe");
 return dir_ == Direction::kGe; }
  inline bool IsGt() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_6(mht_6_v, 244, "", "./tensorflow/compiler/xla/comparison_util.h", "IsGt");
 return dir_ == Direction::kGt; }
  inline bool IsLt() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_7(mht_7_v, 248, "", "./tensorflow/compiler/xla/comparison_util.h", "IsLt");
 return dir_ == Direction::kLt; }
  inline bool IsFloat() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_8(mht_8_v, 252, "", "./tensorflow/compiler/xla/comparison_util.h", "IsFloat");
 return type_ == Type::kFloat; }
  inline bool IsFloatTotalOrder() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_9(mht_9_v, 256, "", "./tensorflow/compiler/xla/comparison_util.h", "IsFloatTotalOrder");

    return type_ == Type::kFloatTotalOrder;
  }
  inline bool IsSigned() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_10(mht_10_v, 262, "", "./tensorflow/compiler/xla/comparison_util.h", "IsSigned");
 return type_ == Type::kSigned; }
  inline bool IsUnsigned() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_11(mht_11_v, 266, "", "./tensorflow/compiler/xla/comparison_util.h", "IsUnsigned");
 return type_ == Type::kUnsigned; }

  // Returns true for comparisons, for which (a dir a) is always true.
  bool IsReflexive() const;

  // Returns true for comparisons, for which (a dir a) is always false.
  bool IsAntireflexive() const;

  // Gets the converse of the given comparison direction (e.g. >= turns to <=).
  // Useful when commuting operands to get constants into
  // immediate-accepting positions in the ISA.
  Comparison Converse() const;

  // Gets the inverse of the given comparison if it exists (e.g. >= turns to <).
  // Returns optional value because not all inversions may be supported.
  absl::optional<Comparison> Inverse() const;

  std::string ToString(std::string prefix1 = ".",
                       std::string prefix2 = ".") const;

  template <typename T>
  std::function<bool(T, T)> GetComparator() const {
    switch (GetDirection()) {
      case Direction::kEq:
        return std::equal_to<T>();
      case Direction::kNe:
        return std::not_equal_to<T>();
      case Direction::kGe:
        return std::greater_equal<T>();
      case Direction::kGt:
        return std::greater<T>();
      case Direction::kLe:
        return std::less_equal<T>();
      case Direction::kLt:
        return std::less<T>();
    }
  }

  template <typename T>
  bool Compare(const T a, const T b) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_12(mht_12_v, 308, "", "./tensorflow/compiler/xla/comparison_util.h", "Compare");

    return GetComparator<T>()(a, b);
  }
  static Type DefaultComparisonType(PrimitiveType t);

 private:
  static Direction Converse(Direction dir);
  static Direction Inverse(Direction dir);

  const Direction dir_;
  Type type_;
};

inline std::ostream& operator<<(std::ostream& os, const Comparison& cmp) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_13(mht_13_v, 324, "", "./tensorflow/compiler/xla/comparison_util.h", "operator<<");

  return os << cmp.ToString();
}
std::string ComparisonDirectionToString(Comparison::Direction direction);
std::string ComparisonTypeToString(Comparison::Type type);

StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction_name);

StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view compare_type_name);

using ComparisonDirection = Comparison::Direction;

// Returns a comparison function using the provided key function on each value,
// i.e. `key_fn(a) < key_fn(b)`.
template <typename KeyFn>
auto LessThanByKey(KeyFn&& key_fn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_14(mht_14_v, 344, "", "./tensorflow/compiler/xla/comparison_util.h", "LessThanByKey");

  return [=](const auto& a, const auto& b) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTh mht_15(mht_15_v, 348, "", "./tensorflow/compiler/xla/comparison_util.h", "lambda");
 return key_fn(a) < key_fn(b); };
}

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_
