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
class MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc() {
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

#include "tensorflow/core/platform/str_util.h"

#include <cctype>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace str_util {

string CEscape(StringPiece src) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/platform/str_util.cc", "CEscape");
 return absl::CEscape(src); }

bool CUnescape(StringPiece source, string* dest, string* error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/platform/str_util.cc", "CUnescape");

  return absl::CUnescape(source, dest, error);
}

void StripTrailingWhitespace(string* s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/platform/str_util.cc", "StripTrailingWhitespace");

  absl::StripTrailingAsciiWhitespace(s);
}

size_t RemoveLeadingWhitespace(StringPiece* text) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/platform/str_util.cc", "RemoveLeadingWhitespace");

  absl::string_view new_text = absl::StripLeadingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

size_t RemoveTrailingWhitespace(StringPiece* text) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_4(mht_4_v, 230, "", "./tensorflow/core/platform/str_util.cc", "RemoveTrailingWhitespace");

  absl::string_view new_text = absl::StripTrailingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

size_t RemoveWhitespaceContext(StringPiece* text) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_5(mht_5_v, 240, "", "./tensorflow/core/platform/str_util.cc", "RemoveWhitespaceContext");

  absl::string_view new_text = absl::StripAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

bool ConsumeLeadingDigits(StringPiece* s, uint64* val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_6(mht_6_v, 250, "", "./tensorflow/core/platform/str_util.cc", "ConsumeLeadingDigits");

  const char* p = s->data();
  const char* limit = p + s->size();
  uint64 v = 0;
  while (p < limit) {
    const char c = *p;
    if (c < '0' || c > '9') break;
    uint64 new_v = (v * 10) + (c - '0');
    if (new_v / 8 < v) {
      // Overflow occurred
      return false;
    }
    v = new_v;
    p++;
  }
  if (p > s->data()) {
    // Consume some digits
    s->remove_prefix(p - s->data());
    *val = v;
    return true;
  } else {
    return false;
  }
}

bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/platform/str_util.cc", "ConsumeNonWhitespace");

  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (isspace(c)) break;
    p++;
  }
  const size_t n = p - s->data();
  if (n > 0) {
    *val = StringPiece(s->data(), n);
    s->remove_prefix(n);
    return true;
  } else {
    *val = StringPiece();
    return false;
  }
}

bool ConsumePrefix(StringPiece* s, StringPiece expected) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/platform/str_util.cc", "ConsumePrefix");

  return absl::ConsumePrefix(s, expected);
}

bool ConsumeSuffix(StringPiece* s, StringPiece expected) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_9(mht_9_v, 307, "", "./tensorflow/core/platform/str_util.cc", "ConsumeSuffix");

  return absl::ConsumeSuffix(s, expected);
}

StringPiece StripPrefix(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_10(mht_10_v, 314, "", "./tensorflow/core/platform/str_util.cc", "StripPrefix");

  return absl::StripPrefix(s, expected);
}

StringPiece StripSuffix(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_11(mht_11_v, 321, "", "./tensorflow/core/platform/str_util.cc", "StripSuffix");

  return absl::StripSuffix(s, expected);
}

// Return lower-cased version of s.
string Lowercase(StringPiece s) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_12(mht_12_v, 329, "", "./tensorflow/core/platform/str_util.cc", "Lowercase");
 return absl::AsciiStrToLower(s); }

// Return upper-cased version of s.
string Uppercase(StringPiece s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_13(mht_13_v, 335, "", "./tensorflow/core/platform/str_util.cc", "Uppercase");
 return absl::AsciiStrToUpper(s); }

void TitlecaseString(string* s, StringPiece delimiters) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_14(mht_14_v, 340, "", "./tensorflow/core/platform/str_util.cc", "TitlecaseString");

  bool upper = true;
  for (string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = toupper(*ss);
    }
    upper = (delimiters.find(*ss) != StringPiece::npos);
  }
}

string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                     bool replace_all) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_15(mht_15_v, 354, "", "./tensorflow/core/platform/str_util.cc", "StringReplace");

  // TODO(jlebar): We could avoid having to shift data around in the string if
  // we had a StringPiece::find() overload that searched for a StringPiece.
  string res(s);
  size_t pos = 0;
  while ((pos = res.find(oldsub.data(), pos, oldsub.size())) != string::npos) {
    res.replace(pos, oldsub.size(), newsub.data(), newsub.size());
    pos += newsub.size();
    if (oldsub.empty()) {
      pos++;  // Match at the beginning of the text and after every byte
    }
    if (!replace_all) {
      break;
    }
  }
  return res;
}

bool StartsWith(StringPiece text, StringPiece prefix) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_16(mht_16_v, 375, "", "./tensorflow/core/platform/str_util.cc", "StartsWith");

  return absl::StartsWith(text, prefix);
}

bool EndsWith(StringPiece text, StringPiece suffix) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_17(mht_17_v, 382, "", "./tensorflow/core/platform/str_util.cc", "EndsWith");

  return absl::EndsWith(text, suffix);
}

bool StrContains(StringPiece haystack, StringPiece needle) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_18(mht_18_v, 389, "", "./tensorflow/core/platform/str_util.cc", "StrContains");

  return absl::StrContains(haystack, needle);
}

size_t Strnlen(const char* str, const size_t string_max_len) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_19(mht_19_v, 397, "", "./tensorflow/core/platform/str_util.cc", "Strnlen");

  size_t len = 0;
  while (len < string_max_len && str[len] != '\0') {
    ++len;
  }
  return len;
}

string ArgDefCase(StringPiece s) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_utilDTcc mht_20(mht_20_v, 408, "", "./tensorflow/core/platform/str_util.cc", "ArgDefCase");

  const size_t n = s.size();

  // Compute the size of resulting string.
  // Number of extra underscores we will need to add.
  size_t extra_us = 0;
  // Number of non-alpha chars in the beginning to skip.
  size_t to_skip = 0;
  for (size_t i = 0; i < n; ++i) {
    // If we are skipping and current letter is non-alpha, skip it as well
    if (i == to_skip && !isalpha(s[i])) {
      ++to_skip;
      continue;
    }

    // If we are here, we are not skipping any more.
    // If this letter is upper case, not the very first char in the
    // resulting string, and previous letter isn't replaced with an underscore,
    // we will need to insert an underscore.
    if (isupper(s[i]) && i != to_skip && i > 0 && isalnum(s[i - 1])) {
      ++extra_us;
    }
  }

  // Initialize result with all '_'s. There is no string
  // constructor that does not initialize memory.
  string result(n + extra_us - to_skip, '_');
  // i - index into s
  // j - index into result
  for (size_t i = to_skip, j = 0; i < n; ++i, ++j) {
    DCHECK_LT(j, result.size());
    char c = s[i];
    // If c is not alphanumeric, we don't need to do anything
    // since there is already an underscore in its place.
    if (isalnum(c)) {
      if (isupper(c)) {
        // If current char is upper case, we might need to insert an
        // underscore.
        if (i != to_skip) {
          DCHECK_GT(j, 0);
          if (result[j - 1] != '_') ++j;
        }
        result[j] = tolower(c);
      } else {
        result[j] = c;
      }
    }
  }

  return result;
}

}  // namespace str_util
}  // namespace tensorflow
