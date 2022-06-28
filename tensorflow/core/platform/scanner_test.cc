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
class MHTracer_DTPStensorflowPScorePSplatformPSscanner_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSscanner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSscanner_testDTcc() {
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

#include "tensorflow/core/platform/scanner.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace strings {

class ScannerTest : public ::testing::Test {
 protected:
  // Returns a string with all chars that are in <clz>, in byte value order.
  string ClassStr(Scanner::CharClass clz) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSscanner_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/platform/scanner_test.cc", "ClassStr");

    string s;
    for (int i = 0; i < 256; ++i) {
      char ch = i;
      if (Scanner::Matches(clz, ch)) {
        s += ch;
      }
    }
    return s;
  }
};

TEST_F(ScannerTest, Any) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("   horse0123")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("   horse", match);
  EXPECT_EQ("0123", remaining);

  EXPECT_TRUE(Scanner("")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);

  EXPECT_TRUE(Scanner("----")
                  .Any(Scanner::SPACE)
                  .Any(Scanner::DIGIT)
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("----", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, AnySpace) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("  a b ")
                  .AnySpace()
                  .One(Scanner::LETTER)
                  .AnySpace()
                  .GetResult(&remaining, &match));
  EXPECT_EQ("  a ", match);
  EXPECT_EQ("b ", remaining);
}

TEST_F(ScannerTest, AnyEscapedNewline) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("\\\n")
                  .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("\\\n", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, AnyEmptyString) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("")
                  .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);
}

TEST_F(ScannerTest, Eos) {
  EXPECT_FALSE(Scanner("a").Eos().GetResult());
  EXPECT_TRUE(Scanner("").Eos().GetResult());
  EXPECT_FALSE(Scanner("abc").OneLiteral("ab").Eos().GetResult());
  EXPECT_TRUE(Scanner("abc").OneLiteral("abc").Eos().GetResult());
}

TEST_F(ScannerTest, Many) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("abc").Many(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("0").Many(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("").Many(Scanner::LETTER).GetResult());

  EXPECT_TRUE(
      Scanner("abc ").Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ(" ", remaining);
  EXPECT_EQ("abc", match);
  EXPECT_TRUE(
      Scanner("abc").Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("abc", match);
}

TEST_F(ScannerTest, One) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner("abc").One(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("0").One(Scanner::LETTER).GetResult());
  EXPECT_FALSE(Scanner("").One(Scanner::LETTER).GetResult());

  EXPECT_TRUE(Scanner("abc")
                  .One(Scanner::LETTER)
                  .One(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("c", remaining);
  EXPECT_EQ("ab", match);
  EXPECT_TRUE(Scanner("a").One(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("a", match);
}

TEST_F(ScannerTest, OneLiteral) {
  EXPECT_FALSE(Scanner("abc").OneLiteral("abC").GetResult());
  EXPECT_TRUE(Scanner("abc").OneLiteral("ab").OneLiteral("c").GetResult());
}

TEST_F(ScannerTest, ScanUntil) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner(R"(' \1 \2 \3 \' \\'rest)")
                  .OneLiteral("'")
                  .ScanUntil('\'')
                  .OneLiteral("'")
                  .GetResult(&remaining, &match));
  EXPECT_EQ(R"( \\'rest)", remaining);
  EXPECT_EQ(R"(' \1 \2 \3 \')", match);

  // The "scan until" character is not present.
  remaining = match = "unset";
  EXPECT_FALSE(Scanner(R"(' \1 \2 \3 \\rest)")
                   .OneLiteral("'")
                   .ScanUntil('\'')
                   .GetResult(&remaining, &match));
  EXPECT_EQ("unset", remaining);
  EXPECT_EQ("unset", match);

  // Scan until an escape character.
  remaining = match = "";
  EXPECT_TRUE(
      Scanner(R"(123\456)").ScanUntil('\\').GetResult(&remaining, &match));
  EXPECT_EQ(R"(\456)", remaining);
  EXPECT_EQ("123", match);
}

TEST_F(ScannerTest, ScanEscapedUntil) {
  StringPiece remaining, match;
  EXPECT_TRUE(Scanner(R"(' \1 \2 \3 \' \\'rest)")
                  .OneLiteral("'")
                  .ScanEscapedUntil('\'')
                  .OneLiteral("'")
                  .GetResult(&remaining, &match));
  EXPECT_EQ("rest", remaining);
  EXPECT_EQ(R"(' \1 \2 \3 \' \\')", match);

  // The "scan until" character is not present.
  remaining = match = "unset";
  EXPECT_FALSE(Scanner(R"(' \1 \2 \3 \' \\rest)")
                   .OneLiteral("'")
                   .ScanEscapedUntil('\'')
                   .GetResult(&remaining, &match));
  EXPECT_EQ("unset", remaining);
  EXPECT_EQ("unset", match);
}

TEST_F(ScannerTest, ZeroOrOneLiteral) {
  StringPiece remaining, match;
  EXPECT_TRUE(
      Scanner("abc").ZeroOrOneLiteral("abC").GetResult(&remaining, &match));
  EXPECT_EQ("abc", remaining);
  EXPECT_EQ("", match);

  EXPECT_TRUE(
      Scanner("abcd").ZeroOrOneLiteral("ab").ZeroOrOneLiteral("c").GetResult(
          &remaining, &match));
  EXPECT_EQ("d", remaining);
  EXPECT_EQ("abc", match);

  EXPECT_TRUE(
      Scanner("").ZeroOrOneLiteral("abc").GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("", match);
}

// Test output of GetResult (including the forms with optional params),
// and that it can be called multiple times.
TEST_F(ScannerTest, CaptureAndGetResult) {
  StringPiece remaining, match;

  Scanner scan("  first    second");
  EXPECT_TRUE(scan.Any(Scanner::SPACE)
                  .RestartCapture()
                  .One(Scanner::LETTER)
                  .Any(Scanner::LETTER_DIGIT)
                  .StopCapture()
                  .Any(Scanner::SPACE)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("second", remaining);
  EXPECT_EQ("first", match);
  EXPECT_TRUE(scan.GetResult());
  remaining = "";
  EXPECT_TRUE(scan.GetResult(&remaining));
  EXPECT_EQ("second", remaining);
  remaining = "";
  match = "";
  EXPECT_TRUE(scan.GetResult(&remaining, &match));
  EXPECT_EQ("second", remaining);
  EXPECT_EQ("first", match);

  scan.RestartCapture().One(Scanner::LETTER).One(Scanner::LETTER);
  remaining = "";
  match = "";
  EXPECT_TRUE(scan.GetResult(&remaining, &match));
  EXPECT_EQ("cond", remaining);
  EXPECT_EQ("se", match);
}

// Tests that if StopCapture is not called, then calling GetResult, then
// scanning more, then GetResult again will update the capture.
TEST_F(ScannerTest, MultipleGetResultExtendsCapture) {
  StringPiece remaining, match;

  Scanner scan("one2three");
  EXPECT_TRUE(scan.Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("2three", remaining);
  EXPECT_EQ("one", match);
  EXPECT_TRUE(scan.Many(Scanner::DIGIT).GetResult(&remaining, &match));
  EXPECT_EQ("three", remaining);
  EXPECT_EQ("one2", match);
  EXPECT_TRUE(scan.Many(Scanner::LETTER).GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("one2three", match);
}

TEST_F(ScannerTest, FailedMatchDoesntChangeResult) {
  // A failed match doesn't change pointers passed to GetResult.
  Scanner scan("name");
  StringPiece remaining = "rem";
  StringPiece match = "match";
  EXPECT_FALSE(scan.One(Scanner::SPACE).GetResult(&remaining, &match));
  EXPECT_EQ("rem", remaining);
  EXPECT_EQ("match", match);
}

TEST_F(ScannerTest, DefaultCapturesAll) {
  // If RestartCapture() is not called, the whole string is used.
  Scanner scan("a b");
  StringPiece remaining = "rem";
  StringPiece match = "match";
  EXPECT_TRUE(scan.Any(Scanner::LETTER)
                  .AnySpace()
                  .Any(Scanner::LETTER)
                  .GetResult(&remaining, &match));
  EXPECT_EQ("", remaining);
  EXPECT_EQ("a b", match);
}

TEST_F(ScannerTest, AllCharClasses) {
  EXPECT_EQ(256, ClassStr(Scanner::ALL).size());
  EXPECT_EQ("0123456789", ClassStr(Scanner::DIGIT));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER));
  EXPECT_EQ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT));
  EXPECT_EQ(
      "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_UNDERSCORE));
  EXPECT_EQ(
      "-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_DOT_SLASH));
  EXPECT_EQ(
      "-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
      "abcdefghijklmnopqrstuvwxyz",
      ClassStr(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE));
  EXPECT_EQ(".0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT));
  EXPECT_EQ("+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT_PLUS_MINUS));
  EXPECT_EQ(".0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_DOT_UNDERSCORE));
  EXPECT_EQ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LETTER_DIGIT_UNDERSCORE));
  EXPECT_EQ("abcdefghijklmnopqrstuvwxyz", ClassStr(Scanner::LOWERLETTER));
  EXPECT_EQ("0123456789abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LOWERLETTER_DIGIT));
  EXPECT_EQ("0123456789_abcdefghijklmnopqrstuvwxyz",
            ClassStr(Scanner::LOWERLETTER_DIGIT_UNDERSCORE));
  EXPECT_EQ("123456789", ClassStr(Scanner::NON_ZERO_DIGIT));
  EXPECT_EQ("\t\n\v\f\r ", ClassStr(Scanner::SPACE));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ClassStr(Scanner::UPPERLETTER));
  EXPECT_EQ(">", ClassStr(Scanner::RANGLE));
}

TEST_F(ScannerTest, Peek) {
  EXPECT_EQ('a', Scanner("abc").Peek());
  EXPECT_EQ('a', Scanner("abc").Peek('b'));
  EXPECT_EQ('\0', Scanner("").Peek());
  EXPECT_EQ('z', Scanner("").Peek('z'));
  EXPECT_EQ('A', Scanner("0123A").Any(Scanner::DIGIT).Peek());
  EXPECT_EQ('\0', Scanner("0123A").Any(Scanner::LETTER_DIGIT).Peek());
}

}  // namespace strings
}  // namespace tensorflow
