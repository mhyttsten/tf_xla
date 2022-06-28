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

#ifndef TENSORFLOW_CORE_PLATFORM_SCANNER_H_
#define TENSORFLOW_CORE_PLATFORM_SCANNER_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh() {
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

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace strings {

// Scanner provides simplified string parsing, in which a string is parsed as a
// series of scanning calls (e.g. One, Any, Many, OneLiteral, Eos), and then
// finally GetResult is called. If GetResult returns true, then it also returns
// the remaining characters and any captured substring.
//
// The range to capture can be controlled with RestartCapture and StopCapture;
// by default, all processed characters are captured.
class Scanner {
 public:
  // Classes of characters. Each enum name is to be read as the union of the
  // parts - e.g., class LETTER_DIGIT means the class includes all letters and
  // all digits.
  //
  // LETTER means ascii letter a-zA-Z.
  // DIGIT means ascii digit: 0-9.
  enum CharClass {
    // NOTE: When adding a new CharClass, update the AllCharClasses ScannerTest
    // in scanner_test.cc
    ALL,
    DIGIT,
    LETTER,
    LETTER_DIGIT,
    LETTER_DIGIT_DASH_UNDERSCORE,
    LETTER_DIGIT_DASH_DOT_SLASH,             // SLASH is / only, not backslash
    LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE,  // SLASH is / only, not backslash
    LETTER_DIGIT_DOT,
    LETTER_DIGIT_DOT_PLUS_MINUS,
    LETTER_DIGIT_DOT_UNDERSCORE,
    LETTER_DIGIT_UNDERSCORE,
    LOWERLETTER,
    LOWERLETTER_DIGIT,
    LOWERLETTER_DIGIT_UNDERSCORE,
    NON_ZERO_DIGIT,
    SPACE,
    UPPERLETTER,
    RANGLE,
  };

  explicit Scanner(StringPiece source) : cur_(source) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/platform/scanner.h", "Scanner");
 RestartCapture(); }

  // Consume the next character of the given class from input. If the next
  // character is not in the class, then GetResult will ultimately return false.
  Scanner& One(CharClass clz) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_1(mht_1_v, 242, "", "./tensorflow/core/platform/scanner.h", "One");

    if (cur_.empty() || !Matches(clz, cur_[0])) {
      return Error();
    }
    cur_.remove_prefix(1);
    return *this;
  }

  // Consume the next s.size() characters of the input, if they match <s>. If
  // they don't match <s>, this is a no-op.
  Scanner& ZeroOrOneLiteral(StringPiece s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/platform/scanner.h", "ZeroOrOneLiteral");

    str_util::ConsumePrefix(&cur_, s);
    return *this;
  }

  // Consume the next s.size() characters of the input, if they match <s>. If
  // they don't match <s>, then GetResult will ultimately return false.
  Scanner& OneLiteral(StringPiece s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/platform/scanner.h", "OneLiteral");

    if (!str_util::ConsumePrefix(&cur_, s)) {
      error_ = true;
    }
    return *this;
  }

  // Consume characters from the input as long as they match <clz>. Zero
  // characters is still considered a match, so it will never cause GetResult to
  // return false.
  Scanner& Any(CharClass clz) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_4(mht_4_v, 278, "", "./tensorflow/core/platform/scanner.h", "Any");

    while (!cur_.empty() && Matches(clz, cur_[0])) {
      cur_.remove_prefix(1);
    }
    return *this;
  }

  // Shorthand for One(clz).Any(clz).
  Scanner& Many(CharClass clz) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_5(mht_5_v, 289, "", "./tensorflow/core/platform/scanner.h", "Many");
 return One(clz).Any(clz); }

  // Reset the capture start point.
  //
  // Later, when GetResult is called and if it returns true, the capture
  // returned will start at the position at the time this was called.
  Scanner& RestartCapture() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_6(mht_6_v, 298, "", "./tensorflow/core/platform/scanner.h", "RestartCapture");

    capture_start_ = cur_.data();
    capture_end_ = nullptr;
    return *this;
  }

  // Stop capturing input.
  //
  // Later, when GetResult is called and if it returns true, the capture
  // returned will end at the position at the time this was called.
  Scanner& StopCapture() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_7(mht_7_v, 311, "", "./tensorflow/core/platform/scanner.h", "StopCapture");

    capture_end_ = cur_.data();
    return *this;
  }

  // If not at the input of input, then GetResult will ultimately return false.
  Scanner& Eos() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_8(mht_8_v, 320, "", "./tensorflow/core/platform/scanner.h", "Eos");

    if (!cur_.empty()) error_ = true;
    return *this;
  }

  // Shorthand for Any(SPACE).
  Scanner& AnySpace() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_9(mht_9_v, 329, "", "./tensorflow/core/platform/scanner.h", "AnySpace");
 return Any(SPACE); }

  // This scans input until <end_ch> is reached. <end_ch> is NOT consumed.
  Scanner& ScanUntil(char end_ch) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("end_ch: '" + std::string(1, end_ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_10(mht_10_v, 336, "", "./tensorflow/core/platform/scanner.h", "ScanUntil");

    ScanUntilImpl(end_ch, false);
    return *this;
  }

  // This scans input until <end_ch> is reached. <end_ch> is NOT consumed.
  // Backslash escape sequences are skipped.
  // Used for implementing quoted string scanning.
  Scanner& ScanEscapedUntil(char end_ch) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("end_ch: '" + std::string(1, end_ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_11(mht_11_v, 348, "", "./tensorflow/core/platform/scanner.h", "ScanEscapedUntil");

    ScanUntilImpl(end_ch, true);
    return *this;
  }

  // Return the next character that will be scanned, or <default_value> if there
  // are no more characters to scan.
  // Note that if a scan operation has failed (so GetResult() returns false),
  // then the value of Peek may or may not have advanced since the scan
  // operation that failed.
  char Peek(char default_value = '\0') const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_12(mht_12_v, 361, "", "./tensorflow/core/platform/scanner.h", "Peek");

    return cur_.empty() ? default_value : cur_[0];
  }

  // Returns false if there are no remaining characters to consume.
  int empty() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_13(mht_13_v, 369, "", "./tensorflow/core/platform/scanner.h", "empty");
 return cur_.empty(); }

  // Returns true if the input string successfully matched. When true is
  // returned, the remaining string is returned in <remaining> and the captured
  // string returned in <capture>, if non-NULL.
  bool GetResult(StringPiece* remaining = nullptr,
                 StringPiece* capture = nullptr);

 private:
  void ScanUntilImpl(char end_ch, bool escaped);

  Scanner& Error() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_14(mht_14_v, 383, "", "./tensorflow/core/platform/scanner.h", "Error");

    error_ = true;
    return *this;
  }

  static bool IsLetter(char ch) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_15(mht_15_v, 392, "", "./tensorflow/core/platform/scanner.h", "IsLetter");

    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
  }

  static bool IsLowerLetter(char ch) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_16(mht_16_v, 400, "", "./tensorflow/core/platform/scanner.h", "IsLowerLetter");
 return ch >= 'a' && ch <= 'z'; }

  static bool IsDigit(char ch) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_17(mht_17_v, 406, "", "./tensorflow/core/platform/scanner.h", "IsDigit");
 return ch >= '0' && ch <= '9'; }

  static bool IsSpace(char ch) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_18(mht_18_v, 412, "", "./tensorflow/core/platform/scanner.h", "IsSpace");

    return (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' ||
            ch == '\r');
  }

  static bool Matches(CharClass clz, char ch) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSscannerDTh mht_19(mht_19_v, 421, "", "./tensorflow/core/platform/scanner.h", "Matches");

    switch (clz) {
      case ALL:
        return true;
      case DIGIT:
        return IsDigit(ch);
      case LETTER:
        return IsLetter(ch);
      case LETTER_DIGIT:
        return IsLetter(ch) || IsDigit(ch);
      case LETTER_DIGIT_DASH_UNDERSCORE:
        return (IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '_');
      case LETTER_DIGIT_DASH_DOT_SLASH:
        return IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '.' ||
               ch == '/';
      case LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE:
        return (IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '.' ||
                ch == '/' || ch == '_');
      case LETTER_DIGIT_DOT:
        return IsLetter(ch) || IsDigit(ch) || ch == '.';
      case LETTER_DIGIT_DOT_PLUS_MINUS:
        return IsLetter(ch) || IsDigit(ch) || ch == '+' || ch == '-' ||
               ch == '.';
      case LETTER_DIGIT_DOT_UNDERSCORE:
        return IsLetter(ch) || IsDigit(ch) || ch == '.' || ch == '_';
      case LETTER_DIGIT_UNDERSCORE:
        return IsLetter(ch) || IsDigit(ch) || ch == '_';
      case LOWERLETTER:
        return ch >= 'a' && ch <= 'z';
      case LOWERLETTER_DIGIT:
        return IsLowerLetter(ch) || IsDigit(ch);
      case LOWERLETTER_DIGIT_UNDERSCORE:
        return IsLowerLetter(ch) || IsDigit(ch) || ch == '_';
      case NON_ZERO_DIGIT:
        return IsDigit(ch) && ch != '0';
      case SPACE:
        return IsSpace(ch);
      case UPPERLETTER:
        return ch >= 'A' && ch <= 'Z';
      case RANGLE:
        return ch == '>';
    }
    return false;
  }

  StringPiece cur_;
  const char* capture_start_ = nullptr;
  const char* capture_end_ = nullptr;
  bool error_ = false;

  friend class ScannerTest;
  TF_DISALLOW_COPY_AND_ASSIGN(Scanner);
};

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_SCANNER_H_
