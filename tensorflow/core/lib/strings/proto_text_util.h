/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
#define TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh() {
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


#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace strings {

static constexpr char kColonSeparator[] = ": ";

// Helper functions for writing proto-text output.
// Used by the code generated from tools/proto_text/gen_proto_text_lib.cc.
class ProtoTextOutput {
 public:
  // Construct a ProtoTextOutput that writes to <output> If short_debug is true,
  // outputs text to match proto.ShortDebugString(); else matches
  // proto.DebugString().
  ProtoTextOutput(string* output, bool short_debug)
      : output_(output),
        short_debug_(short_debug),
        field_separator_(short_debug ? " " : "\n") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/lib/strings/proto_text_util.h", "ProtoTextOutput");
}

  // Writes opening of nested message and increases indent level.
  void OpenNestedMessage(const char field_name[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/lib/strings/proto_text_util.h", "OpenNestedMessage");

    StrAppend(output_, level_empty_ ? "" : field_separator_, indent_,
              field_name, " {", field_separator_);
    if (!short_debug_) StrAppend(&indent_, "  ");
    level_empty_ = true;
  }

  // Writes close of nested message and decreases indent level.
  void CloseNestedMessage() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_2(mht_2_v, 228, "", "./tensorflow/core/lib/strings/proto_text_util.h", "CloseNestedMessage");

    if (!short_debug_) indent_.resize(indent_.size() - 2);
    StrAppend(output_, level_empty_ ? "" : field_separator_, indent_, "}");
    level_empty_ = false;
  }

  // Print the close of the top-level message that was printed.
  void CloseTopMessage() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_3(mht_3_v, 238, "", "./tensorflow/core/lib/strings/proto_text_util.h", "CloseTopMessage");

    if (!short_debug_ && !level_empty_) StrAppend(output_, "\n");
  }

  // Appends a numeric value, like my_field: 123
  template <typename T>
  void AppendNumeric(const char field_name[], T value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_4(mht_4_v, 247, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendNumeric");

    AppendFieldAndValue(field_name, StrCat(value));
  }

  // Appends a numeric value, like my_field: 123, but only if value != 0.
  template <typename T>
  void AppendNumericIfNotZero(const char field_name[], T value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_5(mht_5_v, 256, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendNumericIfNotZero");

    if (value != 0) AppendNumeric(field_name, value);
  }

  // Appends a bool value, either my_field: true or my_field: false.
  void AppendBool(const char field_name[], bool value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_6(mht_6_v, 264, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendBool");

    AppendFieldAndValue(field_name, value ? "true" : "false");
  }

  // Appends a bool value, as my_field: true, only if value is true.
  void AppendBoolIfTrue(const char field_name[], bool value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_7(mht_7_v, 272, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendBoolIfTrue");

    if (value) AppendBool(field_name, value);
  }

  // Appends a string value, like my_field: "abc123".
  void AppendString(const char field_name[], const string& value) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_8(mht_8_v, 281, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendString");

    AppendFieldAndValue(
        field_name, StrCat("\"", ::tensorflow::str_util::CEscape(value), "\""));
  }

  // Appends a string value, like my_field: "abc123", but only if value is not
  // empty.
  void AppendStringIfNotEmpty(const char field_name[], const string& value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_9(mht_9_v, 292, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendStringIfNotEmpty");

    if (!value.empty()) AppendString(field_name, value);
  }

  // Appends the string name of an enum, like my_field: FIRST_ENUM.
  void AppendEnumName(const char field_name[], const string& name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_10(mht_10_v, 301, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendEnumName");

    AppendFieldAndValue(field_name, name);
  }

 private:
  void AppendFieldAndValue(const char field_name[], StringPiece value_text) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_11(mht_11_v, 309, "", "./tensorflow/core/lib/strings/proto_text_util.h", "AppendFieldAndValue");

    absl::StrAppend(output_, level_empty_ ? "" : field_separator_, indent_,
                    field_name, kColonSeparator, value_text);
    level_empty_ = false;
  }

  string* const output_;
  const bool short_debug_;
  const string field_separator_;
  string indent_;

  // False when at least one field has been output for the message at the
  // current deepest level of nesting.
  bool level_empty_ = true;

  TF_DISALLOW_COPY_AND_ASSIGN(ProtoTextOutput);
};

inline void ProtoSpaceAndComments(Scanner* scanner) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_12(mht_12_v, 330, "", "./tensorflow/core/lib/strings/proto_text_util.h", "ProtoSpaceAndComments");

  for (;;) {
    scanner->AnySpace();
    if (scanner->Peek() != '#') return;
    // Skip until newline.
    while (scanner->Peek('\n') != '\n') scanner->One(Scanner::ALL);
  }
}

// Parse the next numeric value from <scanner>, returning false if parsing
// failed.
template <typename T>
bool ProtoParseNumericFromScanner(Scanner* scanner, T* value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_text_utilDTh mht_13(mht_13_v, 345, "", "./tensorflow/core/lib/strings/proto_text_util.h", "ProtoParseNumericFromScanner");

  StringPiece numeric_str;
  scanner->RestartCapture();
  if (!scanner->Many(Scanner::LETTER_DIGIT_DOT_PLUS_MINUS)
           .GetResult(nullptr, &numeric_str)) {
    return false;
  }

  // Special case to disallow multiple leading zeroes, to match proto parsing.
  int leading_zero = 0;
  for (size_t i = 0; i < numeric_str.size(); ++i) {
    const char ch = numeric_str[i];
    if (ch == '0') {
      if (++leading_zero > 1) return false;
    } else if (ch != '-') {
      break;
    }
  }

  ProtoSpaceAndComments(scanner);
  return SafeStringToNumeric<T>(numeric_str, value);
}

// Parse the next boolean value from <scanner>, returning false if parsing
// failed.
bool ProtoParseBoolFromScanner(Scanner* scanner, bool* value);

// Parse the next string literal from <scanner>, returning false if parsing
// failed.
bool ProtoParseStringLiteralFromScanner(Scanner* scanner, string* value);

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
