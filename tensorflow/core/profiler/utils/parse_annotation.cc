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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSparse_annotationDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSparse_annotationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSparse_annotationDTcc() {
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
#include "tensorflow/core/profiler/utils/parse_annotation.h"

#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

std::vector<absl::string_view> SplitNameAndMetadata(
    absl::string_view annotation) {
  std::vector<absl::string_view> parts;
  if (!HasMetadata(annotation)) {
    parts.emplace_back(annotation);
  } else {
    annotation.remove_suffix(1);
    parts = absl::StrSplit(annotation, '#');
    if (parts.size() > 2) {
      parts.resize(2);
    }
  }
  while (parts.size() < 2) {
    parts.emplace_back();
  }
  return parts;
}

// Use comma as separate to split input metadata. However, treat comma inside
// ""/''/[]/{}/() pairs as normal characters.
std::vector<absl::string_view> SplitPairs(absl::string_view metadata) {
  std::vector<absl::string_view> key_value_pairs;
  std::stack<char> quotes;
  size_t start = 0, end = 0;
  for (; end < metadata.size(); ++end) {
    char ch = metadata[end];
    switch (ch) {
      case '\"':
      case '\'':
        if (quotes.empty() || quotes.top() != ch) {
          quotes.push(ch);
        } else {
          quotes.pop();
        }
        break;
      case '{':
      case '(':
      case '[':
        quotes.push(ch);
        break;
      case '}':
        if (!quotes.empty() && quotes.top() == '{') {
          quotes.pop();
        }
        break;
      case ')':
        if (!quotes.empty() && quotes.top() == '(') {
          quotes.pop();
        }
        break;
      case ']':
        if (!quotes.empty() && quotes.top() == '[') {
          quotes.pop();
        }
        break;
      case ',':
        if (quotes.empty()) {
          if (end - start > 1) {
            key_value_pairs.emplace_back(metadata.data() + start, end - start);
          }
          start = end + 1;  // Skip the current ','.
        }
        break;
    }
  }
  if (end - start > 1) {
    key_value_pairs.emplace_back(metadata.data() + start, end - start);
  }
  return key_value_pairs;
}

std::vector<std::pair<absl::string_view, absl::string_view>> ParseMetadata(
    absl::string_view metadata) {
  std::vector<std::pair<absl::string_view, absl::string_view>> key_values;
  for (absl::string_view pair : SplitPairs(metadata)) {
    std::vector<absl::string_view> parts =
        absl::StrSplit(pair, absl::MaxSplits('=', 1));
    if (parts.size() == 2) {
      absl::string_view key = absl::StripAsciiWhitespace(parts[0]);
      absl::string_view value = absl::StripAsciiWhitespace(parts[1]);
      if (!key.empty() && !value.empty()) {
        key_values.push_back({key, value});
      }
    }
  }
  return key_values;
}

}  // namespace

Annotation ParseAnnotation(absl::string_view annotation) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("annotation: \"" + std::string(annotation.data(), annotation.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSparse_annotationDTcc mht_0(mht_0_v, 290, "", "./tensorflow/core/profiler/utils/parse_annotation.cc", "ParseAnnotation");

  Annotation result;
  std::vector<absl::string_view> parts = SplitNameAndMetadata(annotation);
  if (!parts.empty()) {
    result.name = absl::StripAsciiWhitespace(parts[0]);
    for (const auto& key_value : ParseMetadata(parts[1])) {
      result.metadata.push_back({key_value.first, key_value.second});
    }
  }
  return result;
}

std::vector<Annotation> ParseAnnotationStack(
    absl::string_view annotation_stack) {
  std::vector<Annotation> annotations;
  const std::string kAnnotationDelimiter = "::";
  for (absl::string_view annotation : absl::StrSplit(
           annotation_stack, kAnnotationDelimiter, absl::SkipEmpty())) {
    annotations.emplace_back(ParseAnnotation(annotation));
  }
  return annotations;
}

}  // namespace profiler
}  // namespace tensorflow
