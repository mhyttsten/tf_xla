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
class MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc() {
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

#include "tensorflow/compiler/xla/comparison_util.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

std::string ComparisonDirectionToString(Comparison::Direction direction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/comparison_util.cc", "ComparisonDirectionToString");

  switch (direction) {
    case Comparison::Direction::kEq:
      return "EQ";
    case Comparison::Direction::kNe:
      return "NE";
    case Comparison::Direction::kGe:
      return "GE";
    case Comparison::Direction::kGt:
      return "GT";
    case Comparison::Direction::kLe:
      return "LE";
    case Comparison::Direction::kLt:
      return "LT";
    default:
      LOG(FATAL) << "Attempted to print uninitialized comparison direction";
  }
}

StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction_name) {
  static auto* direction_map =
      new absl::flat_hash_map<std::string, Comparison::Direction>({
          {"EQ", Comparison::Direction::kEq},
          {"NE", Comparison::Direction::kNe},
          {"GE", Comparison::Direction::kGe},
          {"GT", Comparison::Direction::kGt},
          {"LE", Comparison::Direction::kLe},
          {"LT", Comparison::Direction::kLt},
      });
  auto it = direction_map->find(direction_name);
  if (it == direction_map->end()) {
    return InvalidArgument("Unknown comparison direction: %s", direction_name);
  }
  return it->second;
}

StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view compare_type_name) {
  static auto* type_map =
      new absl::flat_hash_map<std::string, Comparison::Type>({
          {"FLOAT", Comparison::Type::kFloat},
          {"TOTALORDER", Comparison::Type::kFloatTotalOrder},
          {"SIGNED", Comparison::Type::kSigned},
          {"UNSIGNED", Comparison::Type::kUnsigned},
      });
  auto it = type_map->find(compare_type_name);
  if (it == type_map->end()) {
    return InvalidArgument("Unknown comparison type: %s", compare_type_name);
  }
  return it->second;
}

std::string ComparisonTypeToString(Comparison::Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/xla/comparison_util.cc", "ComparisonTypeToString");

  switch (type) {
    case Comparison::Type::kFloat:
      return "FLOAT";
    case Comparison::Type::kFloatTotalOrder:
      return "TOTALORDER";
    case Comparison::Type::kSigned:
      return "SIGNED";
    case Comparison::Type::kUnsigned:
      return "UNSIGNED";
  }
}

Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir), type_(DefaultComparisonType(type)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::Comparison");
}

Comparison::Type Comparison::DefaultComparisonType(PrimitiveType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_3(mht_3_v, 273, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::DefaultComparisonType");

  switch (type) {
    case S8:
    case S16:
    case S32:
    case S64:
      return Type::kSigned;
    case PRED:
    case U8:
    case U16:
    case U32:
    case U64:
      return Type::kUnsigned;
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
      return Type::kFloat;
    default:
      LOG(FATAL) << "Unsupported comparison mode: " << PrimitiveType_Name(type);
  }
}

Comparison Comparison::Converse() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_4(mht_4_v, 301, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::Converse");

  return Comparison(Converse(dir_), type_);
}

absl::optional<Comparison> Comparison::Inverse() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_5(mht_5_v, 308, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::Inverse");

  switch (type_) {
    case Type::kFloat:
      // Floating-point comparisons don't have inverses unless total order is
      // supported (e.g. comparison can return true if one operand is NaN).
      return absl::nullopt;
    case Type::kFloatTotalOrder:
    case Type::kSigned:
    case Type::kUnsigned:
      return Comparison(Inverse(dir_), type_);
  }
}

bool Comparison::IsReflexive() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_6(mht_6_v, 324, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::IsReflexive");

  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsSigned() || IsUnsigned() || IsFloatTotalOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}

bool Comparison::IsAntireflexive() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_7(mht_7_v, 340, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::IsAntireflexive");

  switch (dir_) {
    case Direction::kNe:
      return IsSigned() || IsUnsigned() || IsFloatTotalOrder();
    case Direction::kGt:
    case Direction::kLt:
      return true;
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return false;
  }
}

/* static */ Comparison::Direction Comparison::Converse(
    Comparison::Direction dir) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_8(mht_8_v, 358, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::Converse");

  switch (dir) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGe;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGt;
  }
}

/* static */ Comparison::Direction Comparison::Inverse(
    Comparison::Direction dir) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_9(mht_9_v, 379, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::Inverse");

  switch (dir) {
    case Direction::kEq:
      return Direction::kNe;
    case Direction::kNe:
      return Direction::kEq;
    case Direction::kGe:
      return Direction::kLt;
    case Direction::kGt:
      return Direction::kLe;
    case Direction::kLe:
      return Direction::kGt;
    case Direction::kLt:
      return Direction::kGe;
  }
}

std::string Comparison::ToString(std::string prefix1,
                                 std::string prefix2) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("prefix1: \"" + prefix1 + "\"");
   mht_10_v.push_back("prefix2: \"" + prefix2 + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPScomparison_utilDTcc mht_10(mht_10_v, 402, "", "./tensorflow/compiler/xla/comparison_util.cc", "Comparison::ToString");

  return absl::StrCat(prefix1, ComparisonDirectionToString(dir_), prefix2,
                      ComparisonTypeToString(type_));
}
}  // namespace xla
