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
class MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc() {
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

#include <vector>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CEscape, Basic) {
  EXPECT_EQ(str_util::CEscape("hello"), "hello");
  EXPECT_EQ(str_util::CEscape("hello\n"), "hello\\n");
  EXPECT_EQ(str_util::CEscape("hello\r"), "hello\\r");
  EXPECT_EQ(str_util::CEscape("\t\r\"'"), "\\t\\r\\\"\\'");
  EXPECT_EQ(str_util::CEscape("\320hi\200"), "\\320hi\\200");
}

string ExpectCUnescapeSuccess(StringPiece source) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/platform/str_util_test.cc", "ExpectCUnescapeSuccess");

  string dest;
  string error;
  EXPECT_TRUE(str_util::CUnescape(source, &dest, &error)) << error;
  return dest;
}

TEST(CUnescape, Basic) {
  EXPECT_EQ("hello", ExpectCUnescapeSuccess("hello"));
  EXPECT_EQ("hello\n", ExpectCUnescapeSuccess("hello\\n"));
  EXPECT_EQ("hello\r", ExpectCUnescapeSuccess("hello\\r"));
  EXPECT_EQ("\t\r\"'", ExpectCUnescapeSuccess("\\t\\r\\\"\\'"));
  EXPECT_EQ("\320hi\200", ExpectCUnescapeSuccess("\\320hi\\200"));
}

TEST(CUnescape, HandlesCopyOnWriteStrings) {
  string dest = "hello";
  string read = dest;
  // For std::string, read and dest now share the same buffer.

  string error;
  StringPiece source = "llohe";
  // CUnescape is going to write "llohe" to dest, so dest's buffer will be
  // reallocated, and read's buffer remains untouched.
  EXPECT_TRUE(str_util::CUnescape(source, &dest, &error));
  EXPECT_EQ("hello", read);
}

TEST(StripTrailingWhitespace, Basic) {
  string test;
  test = "hello";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "hello");

  test = "foo  ";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "foo");

  test = "   ";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "");

  test = "";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, "");

  test = " abc\t";
  str_util::StripTrailingWhitespace(&test);
  EXPECT_EQ(test, " abc");
}

TEST(RemoveLeadingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick\t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 11);
  EXPECT_EQ(data, StringPiece("Quick\t"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece("Quick\t"));
}

TEST(RemoveLeadingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  StringPiece data(text);
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 1);
  EXPECT_EQ(data, StringPiece(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveLeadingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

TEST(RemoveTrailingWhitespace, Basic) {
  string text = "  \t   \n  \r Quick \t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 2);
  EXPECT_EQ(data, StringPiece("  \t   \n  \r Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece("  \t   \n  \r Quick"));
}

TEST(RemoveTrailingWhitespace, TerminationHandling) {
  // check termination handling
  string text = "\t";
  StringPiece data(text);
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 1);
  EXPECT_EQ(data, StringPiece(""));

  // check termination handling again
  EXPECT_EQ(str_util::RemoveTrailingWhitespace(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

TEST(RemoveWhitespaceContext, Basic) {
  string text = "  \t   \n  \r Quick \t";
  StringPiece data(text);
  // check that all whitespace is removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 13);
  EXPECT_EQ(data, StringPiece("Quick"));
  // check that non-whitespace is not removed
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, StringPiece("Quick"));

  // Test empty string
  text = "";
  data = text;
  EXPECT_EQ(str_util::RemoveWhitespaceContext(&data), 0);
  EXPECT_EQ(data, StringPiece(""));
}

void TestConsumeLeadingDigits(StringPiece s, int64_t expected,
                              StringPiece remaining) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc mht_1(mht_1_v, 319, "", "./tensorflow/core/platform/str_util_test.cc", "TestConsumeLeadingDigits");

  uint64 v;
  StringPiece input(s);
  if (str_util::ConsumeLeadingDigits(&input, &v)) {
    EXPECT_EQ(v, static_cast<uint64>(expected));
    EXPECT_EQ(input, remaining);
  } else {
    EXPECT_LT(expected, 0);
    EXPECT_EQ(input, remaining);
  }
}

TEST(ConsumeLeadingDigits, Basic) {
  using str_util::ConsumeLeadingDigits;

  TestConsumeLeadingDigits("123", 123, "");
  TestConsumeLeadingDigits("a123", -1, "a123");
  TestConsumeLeadingDigits("9_", 9, "_");
  TestConsumeLeadingDigits("11111111111xyz", 11111111111ll, "xyz");

  // Overflow case
  TestConsumeLeadingDigits("1111111111111111111111111111111xyz", -1,
                           "1111111111111111111111111111111xyz");

  // 2^64
  TestConsumeLeadingDigits("18446744073709551616xyz", -1,
                           "18446744073709551616xyz");
  // 2^64-1
  TestConsumeLeadingDigits("18446744073709551615xyz", 18446744073709551615ull,
                           "xyz");
  // (2^64-1)*10+9
  TestConsumeLeadingDigits("184467440737095516159yz", -1,
                           "184467440737095516159yz");
}

void TestConsumeNonWhitespace(StringPiece s, StringPiece expected,
                              StringPiece remaining) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc mht_2(mht_2_v, 358, "", "./tensorflow/core/platform/str_util_test.cc", "TestConsumeNonWhitespace");

  StringPiece v;
  StringPiece input(s);
  if (str_util::ConsumeNonWhitespace(&input, &v)) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(input, remaining);
  } else {
    EXPECT_EQ(expected, "");
    EXPECT_EQ(input, remaining);
  }
}

TEST(ConsumeNonWhitespace, Basic) {
  TestConsumeNonWhitespace("", "", "");
  TestConsumeNonWhitespace(" ", "", " ");
  TestConsumeNonWhitespace("abc", "abc", "");
  TestConsumeNonWhitespace("abc ", "abc", " ");
}

TEST(ConsumePrefix, Basic) {
  string s("abcdef");
  StringPiece input(s);
  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abcdefg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abce"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(str_util::ConsumePrefix(&input, ""));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(str_util::ConsumePrefix(&input, "abcdeg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(str_util::ConsumePrefix(&input, "abcdef"));
  EXPECT_EQ(input, "");

  input = s;
  EXPECT_TRUE(str_util::ConsumePrefix(&input, "abcde"));
  EXPECT_EQ(input, "f");
}

TEST(StripPrefix, Basic) {
  EXPECT_EQ(str_util::StripPrefix("abcdef", "abcdefg"), "abcdef");
  EXPECT_EQ(str_util::StripPrefix("abcdef", "abce"), "abcdef");
  EXPECT_EQ(str_util::StripPrefix("abcdef", ""), "abcdef");
  EXPECT_EQ(str_util::StripPrefix("abcdef", "abcdeg"), "abcdef");
  EXPECT_EQ(str_util::StripPrefix("abcdef", "abcdef"), "");
  EXPECT_EQ(str_util::StripPrefix("abcdef", "abcde"), "f");
}

TEST(JoinStrings, Basic) {
  std::vector<string> s;
  s = {"hi"};
  EXPECT_EQ(str_util::Join(s, " "), "hi");
  s = {"hi", "there", "strings"};
  EXPECT_EQ(str_util::Join(s, " "), "hi there strings");

  std::vector<StringPiece> sp;
  sp = {"hi"};
  EXPECT_EQ(str_util::Join(sp, ",,"), "hi");
  sp = {"hi", "there", "strings"};
  EXPECT_EQ(str_util::Join(sp, "--"), "hi--there--strings");
}

TEST(JoinStrings, Join3) {
  std::vector<string> s;
  s = {"hi"};
  auto l1 = [](string* out, string s) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc mht_3(mht_3_v, 430, "", "./tensorflow/core/platform/str_util_test.cc", "lambda");
 *out += s; };
  EXPECT_EQ(str_util::Join(s, " ", l1), "hi");
  s = {"hi", "there", "strings"};
  auto l2 = [](string* out, string s) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstr_util_testDTcc mht_4(mht_4_v, 437, "", "./tensorflow/core/platform/str_util_test.cc", "lambda");
 *out += s[0]; };
  EXPECT_EQ(str_util::Join(s, " ", l2), "h t s");
}

TEST(Split, Basic) {
  EXPECT_TRUE(str_util::Split("", ',').empty());
  EXPECT_EQ(str_util::Join(str_util::Split("a", ','), "|"), "a");
  EXPECT_EQ(str_util::Join(str_util::Split(",", ','), "|"), "|");
  EXPECT_EQ(str_util::Join(str_util::Split("a,b,c", ','), "|"), "a|b|c");
  EXPECT_EQ(str_util::Join(str_util::Split("a,,,b,,c,", ','), "|"),
            "a|||b||c|");
  EXPECT_EQ(str_util::Join(str_util::Split("a!,!b,!c,", ",!"), "|"),
            "a|||b||c|");
  EXPECT_EQ(str_util::Join(
                str_util::Split("a,,,b,,c,", ',', str_util::SkipEmpty()), "|"),
            "a|b|c");
  EXPECT_EQ(
      str_util::Join(
          str_util::Split("a,  ,b,,c,", ',', str_util::SkipWhitespace()), "|"),
      "a|b|c");
  EXPECT_EQ(str_util::Join(str_util::Split("a.  !b,;c,", ".,;!",
                                           str_util::SkipWhitespace()),
                           "|"),
            "a|b|c");
}

TEST(Lowercase, Basic) {
  EXPECT_EQ("", str_util::Lowercase(""));
  EXPECT_EQ("hello", str_util::Lowercase("hello"));
  EXPECT_EQ("hello world", str_util::Lowercase("Hello World"));
}

TEST(Uppercase, Basic) {
  EXPECT_EQ("", str_util::Uppercase(""));
  EXPECT_EQ("HELLO", str_util::Uppercase("hello"));
  EXPECT_EQ("HELLO WORLD", str_util::Uppercase("Hello World"));
}

TEST(SnakeCase, Basic) {
  EXPECT_EQ("", str_util::ArgDefCase(""));
  EXPECT_EQ("", str_util::ArgDefCase("!"));
  EXPECT_EQ("", str_util::ArgDefCase("5"));
  EXPECT_EQ("", str_util::ArgDefCase("!:"));
  EXPECT_EQ("", str_util::ArgDefCase("5-5"));
  EXPECT_EQ("", str_util::ArgDefCase("_!"));
  EXPECT_EQ("", str_util::ArgDefCase("_5"));
  EXPECT_EQ("a", str_util::ArgDefCase("_a"));
  EXPECT_EQ("a", str_util::ArgDefCase("_A"));
  EXPECT_EQ("i", str_util::ArgDefCase("I"));
  EXPECT_EQ("i", str_util::ArgDefCase("i"));
  EXPECT_EQ("i_", str_util::ArgDefCase("I%"));
  EXPECT_EQ("i_", str_util::ArgDefCase("i%"));
  EXPECT_EQ("i", str_util::ArgDefCase("%I"));
  EXPECT_EQ("i", str_util::ArgDefCase("-i"));
  EXPECT_EQ("i", str_util::ArgDefCase("3i"));
  EXPECT_EQ("i", str_util::ArgDefCase("32i"));
  EXPECT_EQ("i3", str_util::ArgDefCase("i3"));
  EXPECT_EQ("i_a3", str_util::ArgDefCase("i_A3"));
  EXPECT_EQ("i_i", str_util::ArgDefCase("II"));
  EXPECT_EQ("i_i", str_util::ArgDefCase("I_I"));
  EXPECT_EQ("i__i", str_util::ArgDefCase("I__I"));
  EXPECT_EQ("i_i_32", str_util::ArgDefCase("II-32"));
  EXPECT_EQ("ii_32", str_util::ArgDefCase("Ii-32"));
  EXPECT_EQ("hi_there", str_util::ArgDefCase("HiThere"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("Hi!Hi"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("HiHi"));
  EXPECT_EQ("hihi", str_util::ArgDefCase("Hihi"));
  EXPECT_EQ("hi_hi", str_util::ArgDefCase("Hi_Hi"));
}

TEST(TitlecaseString, Basic) {
  string s = "sparse_lookup";
  str_util::TitlecaseString(&s, "_");
  ASSERT_EQ(s, "Sparse_Lookup");

  s = "sparse_lookup";
  str_util::TitlecaseString(&s, " ");
  ASSERT_EQ(s, "Sparse_lookup");

  s = "dense";
  str_util::TitlecaseString(&s, " ");
  ASSERT_EQ(s, "Dense");
}

TEST(StringReplace, Basic) {
  EXPECT_EQ("XYZ_XYZ_XYZ", str_util::StringReplace("ABC_ABC_ABC", "ABC", "XYZ",
                                                   /*replace_all=*/true));
}

TEST(StringReplace, OnlyFirst) {
  EXPECT_EQ("XYZ_ABC_ABC", str_util::StringReplace("ABC_ABC_ABC", "ABC", "XYZ",
                                                   /*replace_all=*/false));
}

TEST(StringReplace, IncreaseLength) {
  EXPECT_EQ("a b c",
            str_util::StringReplace("abc", "b", " b ", /*replace_all=*/true));
}

TEST(StringReplace, IncreaseLengthMultipleMatches) {
  EXPECT_EQ("a b  b c",
            str_util::StringReplace("abbc", "b", " b ", /*replace_all=*/true));
}

TEST(StringReplace, NoChange) {
  EXPECT_EQ("abc",
            str_util::StringReplace("abc", "d", "X", /*replace_all=*/true));
}

TEST(StringReplace, EmptyStringReplaceFirst) {
  EXPECT_EQ("", str_util::StringReplace("", "a", "X", /*replace_all=*/false));
}

TEST(StringReplace, EmptyStringReplaceAll) {
  EXPECT_EQ("", str_util::StringReplace("", "a", "X", /*replace_all=*/true));
}

TEST(Strnlen, Basic) {
  EXPECT_EQ(0, str_util::Strnlen("ab", 0));
  EXPECT_EQ(1, str_util::Strnlen("a", 1));
  EXPECT_EQ(2, str_util::Strnlen("abcd", 2));
  EXPECT_EQ(3, str_util::Strnlen("abc", 10));
  EXPECT_EQ(4, str_util::Strnlen("a \t\n", 10));
}

}  // namespace tensorflow
