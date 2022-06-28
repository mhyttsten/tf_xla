/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
#define TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTh() {
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

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// Basic string utility routines
namespace tensorflow {
namespace str_util {

// Returns a version of 'src' where unprintable characters have been
// escaped using C-style escape sequences.
std::string CEscape(StringPiece src);

// Copies "source" to "dest", rewriting C-style escape sequences --
// '\n', '\r', '\\', '\ooo', etc -- to their ASCII equivalents.
//
// Errors: Sets the description of the first encountered error in
// 'error'. To disable error reporting, set 'error' to NULL.
//
// NOTE: Does not support \u or \U!
bool CUnescape(StringPiece source, std::string* dest, std::string* error);

// Removes any trailing whitespace from "*s".
void StripTrailingWhitespace(std::string* s);

// Removes leading ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveLeadingWhitespace(StringPiece* text);

// Removes trailing ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveTrailingWhitespace(StringPiece* text);

// Removes leading and trailing ascii_isspace() chars.
// Returns number of chars removed.
size_t RemoveWhitespaceContext(StringPiece* text);

// Consume a leading positive integer value.  If any digits were
// found, store the value of the leading unsigned number in "*val",
// advance "*s" past the consumed number, and return true.  If
// overflow occurred, returns false.  Otherwise, returns false.
bool ConsumeLeadingDigits(StringPiece* s, uint64* val);

// Consume a leading token composed of non-whitespace characters only.
// If *s starts with a non-zero number of non-whitespace characters, store
// them in *val, advance *s past them, and return true.  Else return false.
bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val);

// If "*s" starts with "expected", consume it and return true.
// Otherwise, return false.
bool ConsumePrefix(StringPiece* s, StringPiece expected);

// If "*s" ends with "expected", remove it and return true.
// Otherwise, return false.
bool ConsumeSuffix(StringPiece* s, StringPiece expected);

// If "s" starts with "expected", return a view into "s" after "expected" but
// keep "s" unchanged.
// Otherwise, return the original "s".
TF_MUST_USE_RESULT StringPiece StripPrefix(StringPiece s, StringPiece expected);

// If "s" ends with "expected", return a view into "s" until "expected" but
// keep "s" unchanged.
// Otherwise, return the original "s".
TF_MUST_USE_RESULT StringPiece StripSuffix(StringPiece s, StringPiece expected);

// Return lower-cased version of s.
std::string Lowercase(StringPiece s);

// Return upper-cased version of s.
std::string Uppercase(StringPiece s);

// Capitalize first character of each word in "*s".  "delimiters" is a
// set of characters that can be used as word boundaries.
void TitlecaseString(std::string* s, StringPiece delimiters);

// Replaces the first occurrence (if replace_all is false) or all occurrences
// (if replace_all is true) of oldsub in s with newsub.
std::string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                          bool replace_all);

// Join functionality
template <typename T>
std::string Join(const T& s, const char* sep) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("sep: \"" + (sep == nullptr ? std::string("nullptr") : std::string((char*)sep)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTh mht_0(mht_0_v, 276, "", "./tensorflow/core/platform/str_util.h", "Join");

  return absl::StrJoin(s, sep);
}

// A variant of Join where for each element of "s", f(&dest_string, elem)
// is invoked (f is often constructed with a lambda of the form:
//   [](string* result, ElemType elem)
template <typename T, typename Formatter>
std::string Join(const T& s, const char* sep, Formatter f) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("sep: \"" + (sep == nullptr ? std::string("nullptr") : std::string((char*)sep)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTh mht_1(mht_1_v, 288, "", "./tensorflow/core/platform/str_util.h", "Join");

  return absl::StrJoin(s, sep, f);
}

struct AllowEmpty {
  bool operator()(StringPiece sp) const { return true; }
};
struct SkipEmpty {
  bool operator()(StringPiece sp) const { return !sp.empty(); }
};
struct SkipWhitespace {
  bool operator()(StringPiece sp) const {
    return !absl::StripTrailingAsciiWhitespace(sp).empty();
  }
};

// Split strings using any of the supplied delimiters. For example:
// Split("a,b.c,d", ".,") would return {"a", "b", "c", "d"}.
inline std::vector<string> Split(StringPiece text, StringPiece delims) {
  return text.empty() ? std::vector<string>()
                      : absl::StrSplit(text, absl::ByAnyChar(delims));
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, StringPiece delims, Predicate p) {
  return text.empty() ? std::vector<string>()
                      : absl::StrSplit(text, absl::ByAnyChar(delims), p);
}

inline std::vector<string> Split(StringPiece text, char delim) {
  return text.empty() ? std::vector<string>() : absl::StrSplit(text, delim);
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, char delim, Predicate p) {
  return text.empty() ? std::vector<string>() : absl::StrSplit(text, delim, p);
}

// StartsWith()
//
// Returns whether a given string `text` begins with `prefix`.
bool StartsWith(StringPiece text, StringPiece prefix);

// EndsWith()
//
// Returns whether a given string `text` ends with `suffix`.
bool EndsWith(StringPiece text, StringPiece suffix);

// StrContains()
//
// Returns whether a given string `haystack` contains the substring `needle`.
bool StrContains(StringPiece haystack, StringPiece needle);

// Returns the length of the given null-terminated byte string 'str'.
// Returns 'string_max_len' if the null character was not found in the first
// 'string_max_len' bytes of 'str'.
size_t Strnlen(const char* str, const size_t string_max_len);

//   ----- NON STANDARD, TF SPECIFIC METHOD -----
// Converts "^2ILoveYou!" to "i_love_you_". More specifically:
// - converts all non-alphanumeric characters to underscores
// - replaces each occurrence of a capital letter (except the very
//   first character and if there is already an '_' before it) with '_'
//   followed by this letter in lower case
// - Skips leading non-alpha characters
// This method is useful for producing strings matching "[a-z][a-z0-9_]*"
// as required by OpDef.ArgDef.name. The resulting string is either empty or
// matches this regex.
std::string ArgDefCase(StringPiece s);

}  // namespace str_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
