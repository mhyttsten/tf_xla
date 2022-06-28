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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Given input string and a delimiter returns back a substring including
// delimiters. If there was only starting delimiter found, returns single char.
absl::string_view FindInlineBlock(absl::string_view s, char delimiter) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + std::string(s.data(), s.size()) + "\"");
   mht_0_v.push_back("delimiter: '" + std::string(1, delimiter) + "'");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.cc", "FindInlineBlock");

  size_t start = s.find(delimiter);
  if (start != absl::string_view::npos) {
    size_t end = s.find(delimiter, start + 1);
    if (end != std::string::npos) {
      return s.substr(start, end - start + 1);
    }
    // Special case to indicate that we didn't find the end.
    return s.substr(start, 1);
  }
  return s.substr(s.size(), 0);
}

// For the given 's' and its substring 'subs' returns new substring of 's' that
// begins past 'subs'.
absl::string_view PastSubstr(absl::string_view s, absl::string_view subs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + std::string(s.data(), s.size()) + "\"");
   mht_1_v.push_back("subs: \"" + std::string(subs.data(), subs.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.cc", "PastSubstr");

  return s.substr(subs.data() + subs.size() - s.data());
}

}  // namespace

absl::Status TextPreprocessor::Rewrite(const std::string& input,
                                       std::string* output) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessorDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.cc", "TextPreprocessor::Rewrite");

  absl::string_view s = input;
  std::string result;
  while (true) {
    absl::string_view inline_block = FindInlineBlock(s, inline_delimiter_);
    result.append(s.data(), inline_block.data() - s.data());
    if (inline_block.empty()) {
      break;
    }
    if (inline_block.size() == 1) {
      return absl::NotFoundError("Unable to find end of inline block");
    }
    s = PastSubstr(s, inline_block);
    bool processed = false;
    for (auto& rewrite : inline_rewrites_) {
      if (processed) {
        break;
      }
      switch (rewrite->Rewrite(inline_block.substr(1, inline_block.size() - 2),
                               &result)) {
        case RewriteStatus::NOT_RECOGNIZED:
          // try another rewrite.
          break;
        case RewriteStatus::SUCCESS:
          processed = true;
          break;
        case RewriteStatus::ERROR:
          return absl::InternalError(absl::StrCat("Error while rewriting '",
                                                  inline_block, "': ", result));
      }
    }
    if (!processed) {
      if (!keep_unknown_rewrites_) {
        return absl::NotFoundError(absl::StrCat(
            "Didn't find inline rewrite for '", inline_block, "'"));
      }
      absl::StrAppend(&result, inline_block);
    }
  }
  *output = std::move(result);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
