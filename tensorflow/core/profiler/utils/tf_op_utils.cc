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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc() {
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

#include "tensorflow/core/profiler/utils/tf_op_utils.h"

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kIterator = "Iterator";
const absl::string_view kSeparator = "::";
constexpr char kNameScopeSeparator = '/';
constexpr char kOpNameSuffixSeparator = '_';

bool IsInteger(absl::string_view str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IsInteger");

  int64_t unused;
  return absl::SimpleAtoi(str, &unused);
}

// Returns an op type derived from an op name.
absl::string_view DeriveOpType(absl::string_view full_op_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("full_op_name: \"" + std::string(full_op_name.data(), full_op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "DeriveOpType");

  // Use the op name without name scopes and suffix as an op type. A full op
  // name consists of name scopes, an op type, and optionally a numeric suffix
  // (e.g., model/layer/MatMul_1).
  std::vector<absl::string_view> name_scopes_and_op_name =
      absl::StrSplit(full_op_name, kNameScopeSeparator);
  absl::string_view op_name = name_scopes_and_op_name.back();
  std::vector<absl::string_view> op_type_and_maybe_suffix =
      absl::StrSplit(op_name, kOpNameSuffixSeparator);
  absl::string_view maybe_suffix = op_type_and_maybe_suffix.back();
  absl::string_view op_type = op_name;
  if (IsInteger(maybe_suffix)) {
    // NOTE: assuming a numeric suffix is not part of an op type while
    // technically it is allowed.
    op_type = op_name.substr(0, op_name.size() - maybe_suffix.size() - 1);
  }
  return op_type;
}

}  // namespace

const absl::string_view kUnknownOp = "";  // op types are non-empty strings
const absl::string_view kDatasetOp = "Dataset";
const absl::string_view kMemcpyHToDOp = "MemcpyHToD";
const absl::string_view kMemcpyDToHOp = "MemcpyDToH";
const absl::string_view kMemcpyDToDOp = "MemcpyDToD";
const absl::string_view kMemcpyHToHOp = "MemcpyHToH";

bool IsTfOpName(absl::string_view op_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IsTfOpName");

  // TODO(b/177602927): Confirm the naming convention with the TF team.
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
  return RE2::FullMatch(op_name, *kTfOpNameRegEx);
}

bool IsTfOpType(absl::string_view op_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IsTfOpType");

  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};
  return RE2::FullMatch(op_type, *kTfOpTypeRegEx);
}

bool IsJaxOpType(absl::string_view op_type) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_4(mht_4_v, 270, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IsJaxOpType");

  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*"};
  return RE2::FullMatch(op_type, *kJaxOpTypeRegEx);
}

bool IsJaxOpNameAndType(absl::string_view op_name, absl::string_view op_type) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_5_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IsJaxOpNameAndType");

  if (op_name.empty() || !IsJaxOpType(op_type)) return false;
  std::vector<absl::string_view> split_result =
      absl::StrSplit(op_name, kNameScopeSeparator);
  return absl::StrContains(split_result.back(), op_type);
}

TfOp ParseTfOpFullname(absl::string_view tf_op_fullname) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("tf_op_fullname: \"" + std::string(tf_op_fullname.data(), tf_op_fullname.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "ParseTfOpFullname");

  // TF Op names have the format "name:type".
  TfOp tf_op = {Category::kUnknown, tf_op_fullname, kUnknownOp};
  std::vector<absl::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));
  if (parts.size() != 2) {
    // GPU-related Ops that need to be tracked.
    if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
      tf_op.category = Category::kMemcpyHToD;
      tf_op.type = kMemcpyHToDOp;
    } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
      tf_op.category = Category::kMemcpyDToH;
      tf_op.type = kMemcpyDToHOp;
    } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
      tf_op.category = Category::kMemcpyDToD;
      tf_op.type = kMemcpyDToDOp;
    } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
      tf_op.category = Category::kMemcpyHToH;
      tf_op.type = kMemcpyHToHOp;
    }
    // TODO(ckluk): Include the corresponding Ops on TPU.
  } else if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.category = Category::kTfData;
    tf_op.type = kDatasetOp;
  } else if (IsTfOpType(parts[1]) && IsTfOpName(parts[0])) {
    tf_op = {Category::kTensorFlow, parts[0], parts[1]};
  } else if (IsJaxOpType(parts[1])) {
    tf_op = {Category::kJax, parts[0], parts[1]};
  } else if (parts[1].empty()) {
    tf_op = {Category::kTensorFlow, parts[0], DeriveOpType(parts[0])};
  }
  return tf_op;
}

std::vector<absl::string_view> ParseTfNameScopes(absl::string_view tf_op_name) {
  std::vector<absl::string_view> name_scopes =
      absl::StrSplit(tf_op_name, kNameScopeSeparator);
  // The last element is an op name not TF name scope.
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}

std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  return ParseTfNameScopes(tf_op.name);
}

std::string TfOpEventName(const TfOp& tf_op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_7(mht_7_v, 343, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "TfOpEventName");

  std::string event_name;
  if (tf_op.category == Category::kUnknown) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name = std::string(absl::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.category == Category::kTfData) {
    event_name = DatasetOpEventName(tf_op.name);
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}

std::string TfOpEventName(absl::string_view tf_op_fullname) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("tf_op_fullname: \"" + std::string(tf_op_fullname.data(), tf_op_fullname.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_8(mht_8_v, 360, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "TfOpEventName");

  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}

std::string DatasetOpEventName(absl::string_view full_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("full_name: \"" + std::string(full_name.data(), full_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_9(mht_9_v, 368, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "DatasetOpEventName");

  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return absl::StrCat(kIterator, kSeparator, split_result.back());
}

std::string IteratorName(absl::string_view full_name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("full_name: \"" + std::string(full_name.data(), full_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStf_op_utilsDTcc mht_10(mht_10_v, 378, "", "./tensorflow/core/profiler/utils/tf_op_utils.cc", "IteratorName");

  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return std::string(split_result.back());
}

std::vector<absl::string_view> ParseTensorShapes(
    absl::string_view tensor_shapes) {
  absl::ConsumePrefix(&tensor_shapes, "(");
  absl::ConsumeSuffix(&tensor_shapes, ")");
  return absl::StrSplit(tensor_shapes, ';');
}

}  // namespace profiler
}  // namespace tensorflow
