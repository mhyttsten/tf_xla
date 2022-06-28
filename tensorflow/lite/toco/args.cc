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
class MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc() {
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

#include "tensorflow/lite/toco/args.h"
#include "absl/strings/str_split.h"

namespace toco {
namespace {

// Helper class for SplitStructuredLine parsing.
class ClosingSymbolLookup {
 public:
  explicit ClosingSymbolLookup(const char* symbol_pairs)
      : closing_(), valid_closing_() {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("symbol_pairs: \"" + (symbol_pairs == nullptr ? std::string("nullptr") : std::string((char*)symbol_pairs)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/toco/args.cc", "ClosingSymbolLookup");

    // Initialize the opening/closing arrays.
    for (const char* symbol = symbol_pairs; *symbol != 0; ++symbol) {
      unsigned char opening = *symbol;
      ++symbol;
      // If the string ends before the closing character has been found,
      // use the opening character as the closing character.
      unsigned char closing = *symbol != 0 ? *symbol : opening;
      closing_[opening] = closing;
      valid_closing_[closing] = true;
      if (*symbol == 0) break;
    }
  }

  ClosingSymbolLookup(const ClosingSymbolLookup&) = delete;
  ClosingSymbolLookup& operator=(const ClosingSymbolLookup&) = delete;

  // Returns the closing character corresponding to an opening one,
  // or 0 if the argument is not an opening character.
  char GetClosingChar(char opening) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("opening: '" + std::string(1, opening) + "'");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/toco/args.cc", "GetClosingChar");

    return closing_[static_cast<unsigned char>(opening)];
  }

  // Returns true if the argument is a closing character.
  bool IsClosing(char c) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/toco/args.cc", "IsClosing");

    return valid_closing_[static_cast<unsigned char>(c)];
  }

 private:
  // Maps an opening character to its closing. If the entry contains 0,
  // the character is not in the opening set.
  char closing_[256];
  // Valid closing characters.
  bool valid_closing_[256];
};

bool SplitStructuredLine(absl::string_view line, char delimiter,
                         const char* symbol_pairs,
                         std::vector<absl::string_view>* cols) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("line: \"" + std::string(line.data(), line.size()) + "\"");
   mht_3_v.push_back("delimiter: '" + std::string(1, delimiter) + "'");
   mht_3_v.push_back("symbol_pairs: \"" + (symbol_pairs == nullptr ? std::string("nullptr") : std::string((char*)symbol_pairs)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_3(mht_3_v, 248, "", "./tensorflow/lite/toco/args.cc", "SplitStructuredLine");

  ClosingSymbolLookup lookup(symbol_pairs);

  // Stack of symbols expected to close the current opened expressions.
  std::vector<char> expected_to_close;

  ABSL_RAW_CHECK(cols != nullptr, "");
  cols->push_back(line);
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (expected_to_close.empty() && c == delimiter) {
      // We don't have any open expression, this is a valid separator.
      cols->back().remove_suffix(line.size() - i);
      cols->push_back(line.substr(i + 1));
    } else if (!expected_to_close.empty() && c == expected_to_close.back()) {
      // Can we close the currently open expression?
      expected_to_close.pop_back();
    } else if (lookup.GetClosingChar(c)) {
      // If this is an opening symbol, we open a new expression and push
      // the expected closing symbol on the stack.
      expected_to_close.push_back(lookup.GetClosingChar(c));
    } else if (lookup.IsClosing(c)) {
      // Error: mismatched closing symbol.
      return false;
    }
  }
  if (!expected_to_close.empty()) {
    return false;  // Missing closing symbol(s)
  }
  return true;  // Success
}

inline bool TryStripPrefixString(absl::string_view str,
                                 absl::string_view prefix,
                                 std::string* result) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   mht_4_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/toco/args.cc", "TryStripPrefixString");

  bool res = absl::ConsumePrefix(&str, prefix);
  result->assign(str.begin(), str.end());
  return res;
}

inline bool TryStripSuffixString(absl::string_view str,
                                 absl::string_view suffix,
                                 std::string* result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   mht_5_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_5(mht_5_v, 300, "", "./tensorflow/lite/toco/args.cc", "TryStripSuffixString");

  bool res = absl::ConsumeSuffix(&str, suffix);
  result->assign(str.begin(), str.end());
  return res;
}

}  // namespace

bool Arg<toco::IntList>::Parse(std::string text) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_6(mht_6_v, 312, "", "./tensorflow/lite/toco/args.cc", "Arg<toco::IntList>::Parse");

  parsed_value_.elements.clear();
  specified_ = true;
  // absl::StrSplit("") produces {""}, but we need {} on empty input.
  // TODO(aselle): Moved this from elsewhere, but ahentz recommends we could
  // use absl::SplitLeadingDec32Values(text.c_str(), &parsed_values_.elements)
  if (!text.empty()) {
    int32_t element;
    for (absl::string_view part : absl::StrSplit(text, ',')) {
      if (!absl::SimpleAtoi(part, &element)) return false;
      parsed_value_.elements.push_back(element);
    }
  }
  return true;
}

bool Arg<toco::StringMapList>::Parse(std::string text) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTcc mht_7(mht_7_v, 332, "", "./tensorflow/lite/toco/args.cc", "Arg<toco::StringMapList>::Parse");

  parsed_value_.elements.clear();
  specified_ = true;

  if (text.empty()) {
    return true;
  }

  std::vector<absl::string_view> outer_vector;
  absl::string_view text_disposable_copy = text;
  // TODO(aselle): Change argument parsing when absl supports structuredline.
  SplitStructuredLine(text_disposable_copy, ',', "{}", &outer_vector);
  for (const absl::string_view& outer_member_stringpiece : outer_vector) {
    std::string outer_member(outer_member_stringpiece);
    if (outer_member.empty()) {
      continue;
    }
    std::string outer_member_copy = outer_member;
    absl::StripAsciiWhitespace(&outer_member);
    if (!TryStripPrefixString(outer_member, "{", &outer_member)) return false;
    if (!TryStripSuffixString(outer_member, "}", &outer_member)) return false;
    const std::vector<std::string> inner_fields_vector =
        absl::StrSplit(outer_member, ',');

    std::unordered_map<std::string, std::string> element;
    for (const std::string& member_field : inner_fields_vector) {
      std::vector<std::string> outer_member_key_value =
          absl::StrSplit(member_field, ':');
      if (outer_member_key_value.size() != 2) return false;
      std::string& key = outer_member_key_value[0];
      std::string& value = outer_member_key_value[1];
      absl::StripAsciiWhitespace(&key);
      absl::StripAsciiWhitespace(&value);
      if (element.count(key) != 0) return false;
      element[key] = value;
    }
    parsed_value_.elements.push_back(element);
  }
  return true;
}

}  // namespace toco
