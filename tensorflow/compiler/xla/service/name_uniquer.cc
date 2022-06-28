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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc() {
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

#include "tensorflow/compiler/xla/service/name_uniquer.h"

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

bool IsAllowed(char character) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("character: '" + std::string(1, character) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/name_uniquer.cc", "IsAllowed");

  auto c = static_cast<unsigned char>(character);
  return (absl::ascii_isalnum(c) != 0) || c == '_' || c == '.' || c == '-';
}

}  // namespace

NameUniquer::NameUniquer(const std::string& separator) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("separator: \"" + separator + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc mht_1(mht_1_v, 211, "", "./tensorflow/compiler/xla/service/name_uniquer.cc", "NameUniquer::NameUniquer");

  CHECK(absl::c_all_of(separator, IsAllowed))
      << "separator should comprises allowed characters only";
  separator_ = separator;
}

/*static*/ std::string NameUniquer::GetSanitizedName(absl::string_view name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/xla/service/name_uniquer.cc", "NameUniquer::GetSanitizedName");

  if (name.empty()) {
    return "";
  }

  std::string result(name);
  char c = static_cast<unsigned char>(result[0]);
  if (!absl::ascii_isalpha(c) && c != '_') {
    result[0] = '_';
  }
  for (int i = 1, iter_limit = result.length(); i < iter_limit; i++) {
    if (!IsAllowed(result[i])) {
      result[i] = '_';
    }
  }

  // HLO primitive type names (with the exception of 'tuple') are keywords in
  // the HLO text representation and cannot be names, so append an underscore if
  // the name is a primitive type.
  if (primitive_util::IsPrimitiveTypeName(result) && result != "tuple") {
    result += "_";
  }

  if (absl::StartsWith(result, "__") && !absl::StartsWith(result, "__xla_")) {
    // Morph name prefix __ that is not __xla_, to avoid using name prefixes
    // reserved by the backends, such as __llvm_retpoline_ reserved by the LLVM
    // x86 backend.
    result[0] = 'a';
  }

  return result;
}

std::string NameUniquer::GetUniqueName(absl::string_view prefix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSname_uniquerDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/xla/service/name_uniquer.cc", "NameUniquer::GetUniqueName");

  std::string root =
      GetSanitizedName(prefix.empty() ? "name" : std::string(prefix));

  // Strip away numeric suffix (if any). Only recognize separator if it is in
  // the middle of the name.
  bool has_numeric_suffix = false;
  int64_t numeric_suffix = 0;
  size_t separator_index = root.rfind(separator_);
  if (separator_index != std::string::npos && (separator_index > 0) &&
      (separator_index < root.size() - 1)) {
    std::string after_suffix = root.substr(separator_index + 1);
    if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
      has_numeric_suffix = true;
      // Remove numeric suffix from root.
      root = root.substr(0, separator_index);
    } else {
      // absl::SimpleAtoi may modify numeric_suffix even if it returns false.
      numeric_suffix = 0;
    }
  }

  SequentialIdGenerator& id_generator = generated_names_[root];
  numeric_suffix = id_generator.RegisterId(numeric_suffix);
  if (numeric_suffix == 0) {
    return has_numeric_suffix ? absl::StrCat(root, separator_, 0) : root;
  }
  absl::StrAppend(&root, separator_, numeric_suffix);
  return root;
}

}  // namespace xla
