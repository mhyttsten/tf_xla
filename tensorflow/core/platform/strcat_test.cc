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
class MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc() {
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

#include "tensorflow/core/platform/strcat.h"

#include <string>

#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

#ifdef _MSC_VER
// ssize_t is not a standard C++ type.
typedef ptrdiff_t ssize_t;
#endif

namespace tensorflow {
namespace strings {

// Test StrCat of ints and longs of various sizes and signdedness.
TEST(StrCat, Ints) {
  const int16_t s = -1;
  const uint16 us = 2;
  const int i = -3;
  const unsigned int ui = 4;
  const int32_t l = -5;
  const uint32 ul = 6;
  const int64_t ll = -7;
  const uint64 ull = 8;
  const ptrdiff_t ptrdiff = -9;
  const size_t size = 10;
  const ssize_t ssize = -11;
  const intptr_t intptr = -12;
  const uintptr_t uintptr = 13;
  string answer;
  answer = tensorflow::strings::StrCat(s, us);
  EXPECT_EQ(answer, "-12");
  answer = tensorflow::strings::StrCat(i, ui);
  EXPECT_EQ(answer, "-34");
  answer = tensorflow::strings::StrCat(l, ul);
  EXPECT_EQ(answer, "-56");
  answer = tensorflow::strings::StrCat(ll, ull);
  EXPECT_EQ(answer, "-78");
  answer = tensorflow::strings::StrCat(ptrdiff, size);
  EXPECT_EQ(answer, "-910");
  answer = tensorflow::strings::StrCat(ssize, intptr);
  EXPECT_EQ(answer, "-11-12");
  answer = tensorflow::strings::StrCat(uintptr, 0);
  EXPECT_EQ(answer, "130");
}

TEST(StrCat, Floats) {
  const int s = 0;
  const float f = 1.5f;
  const double d = 1.5;
  const bfloat16 bf(1.5f);

  string answer;
  answer = tensorflow::strings::StrCat(s, f);
  EXPECT_EQ(answer, "01.5");
  answer = tensorflow::strings::StrCat(s, d);
  EXPECT_EQ(answer, "01.5");
  answer = tensorflow::strings::StrCat(s, bf);
  EXPECT_EQ(answer, "01.5");
}

TEST(StrCat, Basics) {
  string result;

  string strs[] = {"Hello", "Cruel", "World"};

  StringPiece pieces[] = {"Hello", "Cruel", "World"};

  const char *c_strs[] = {"Hello", "Cruel", "World"};

  int32 i32s[] = {'H', 'C', 'W'};
  uint64 ui64s[] = {12345678910LL, 10987654321LL};

  result = tensorflow::strings::StrCat(false, true, 2, 3);
  EXPECT_EQ(result, "0123");

  result = tensorflow::strings::StrCat(-1);
  EXPECT_EQ(result, "-1");

  result = tensorflow::strings::StrCat(0.5);
  EXPECT_EQ(result, "0.5");

  result = tensorflow::strings::StrCat(strs[1], pieces[2]);
  EXPECT_EQ(result, "CruelWorld");

  result = tensorflow::strings::StrCat(strs[0], ", ", pieces[2]);
  EXPECT_EQ(result, "Hello, World");

  result =
      tensorflow::strings::StrCat(strs[0], ", ", strs[1], " ", strs[2], "!");
  EXPECT_EQ(result, "Hello, Cruel World!");

  result =
      tensorflow::strings::StrCat(pieces[0], ", ", pieces[1], " ", pieces[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result =
      tensorflow::strings::StrCat(c_strs[0], ", ", c_strs[1], " ", c_strs[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result = tensorflow::strings::StrCat("ASCII ", i32s[0], ", ", i32s[1], " ",
                                       i32s[2], "!");
  EXPECT_EQ(result, "ASCII 72, 67 87!");

  result = tensorflow::strings::StrCat(ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result, "12345678910, 10987654321!");

  string one = "1";  // Actually, it's the size of this string that we want; a
                     // 64-bit build distinguishes between size_t and uint64,
                     // even though they're both unsigned 64-bit values.
  result = tensorflow::strings::StrCat("And a ", one.size(), " and a ",
                                       &result[2] - &result[0], " and a ", one,
                                       " 2 3 4", "!");
  EXPECT_EQ(result, "And a 1 and a 2 and a 1 2 3 4!");

  // result = StrCat("Single chars won't compile", '!');
  // result = StrCat("Neither will NULLs", NULL);
  result = tensorflow::strings::StrCat(
      "To output a char by ASCII/numeric value, use +: ", '!' + 0);
  EXPECT_EQ(result, "To output a char by ASCII/numeric value, use +: 33");

  float f = 100000.5;
  result = tensorflow::strings::StrCat("A hundred K and a half is ", f);
  EXPECT_EQ(result, "A hundred K and a half is 100000.5");

  double d = f;
  d *= d;
  result = tensorflow::strings::StrCat("A hundred K and a half squared is ", d);
  EXPECT_EQ(result, "A hundred K and a half squared is 10000100000.25");

  result = tensorflow::strings::StrCat(1, 2, 333, 4444, 55555, 666666, 7777777,
                                       88888888, 999999999);
  EXPECT_EQ(result, "12333444455555666666777777788888888999999999");
}

TEST(StrCat, MaxArgs) {
  string result;
  // Test 10 up to 26 arguments, the current maximum
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a");
  EXPECT_EQ(result, "123456789a");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b");
  EXPECT_EQ(result, "123456789ab");
  result =
      tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c");
  EXPECT_EQ(result, "123456789abc");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d");
  EXPECT_EQ(result, "123456789abcd");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e");
  EXPECT_EQ(result, "123456789abcde");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f");
  EXPECT_EQ(result, "123456789abcdef");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g");
  EXPECT_EQ(result, "123456789abcdefg");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h");
  EXPECT_EQ(result, "123456789abcdefgh");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i");
  EXPECT_EQ(result, "123456789abcdefghi");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j");
  EXPECT_EQ(result, "123456789abcdefghij");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j", "k");
  EXPECT_EQ(result, "123456789abcdefghijk");
  result =
      tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d",
                                  "e", "f", "g", "h", "i", "j", "k", "l");
  EXPECT_EQ(result, "123456789abcdefghijkl");
  result =
      tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d",
                                  "e", "f", "g", "h", "i", "j", "k", "l", "m");
  EXPECT_EQ(result, "123456789abcdefghijklm");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j", "k",
                                       "l", "m", "n");
  EXPECT_EQ(result, "123456789abcdefghijklmn");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j", "k",
                                       "l", "m", "n", "o");
  EXPECT_EQ(result, "123456789abcdefghijklmno");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j", "k",
                                       "l", "m", "n", "o", "p");
  EXPECT_EQ(result, "123456789abcdefghijklmnop");
  result = tensorflow::strings::StrCat(1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                       "d", "e", "f", "g", "h", "i", "j", "k",
                                       "l", "m", "n", "o", "p", "q");
  EXPECT_EQ(result, "123456789abcdefghijklmnopq");
  // No limit thanks to C++11's variadic templates
  result = tensorflow::strings::StrCat(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "a", "b", "c", "d", "e", "f", "g", "h",
      "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
      "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
      "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z");
  EXPECT_EQ(result,
            "12345678910abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

TEST(StrAppend, Basics) {
  string result = "existing text";

  string strs[] = {"Hello", "Cruel", "World"};

  StringPiece pieces[] = {"Hello", "Cruel", "World"};

  const char *c_strs[] = {"Hello", "Cruel", "World"};

  int32 i32s[] = {'H', 'C', 'W'};
  uint64 ui64s[] = {12345678910LL, 10987654321LL};

  string::size_type old_size = result.size();
  tensorflow::strings::StrAppend(&result, strs[0]);
  EXPECT_EQ(result.substr(old_size), "Hello");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, strs[1], pieces[2]);
  EXPECT_EQ(result.substr(old_size), "CruelWorld");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, strs[0], ", ", pieces[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, World");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, strs[0], ", ", strs[1], " ", strs[2],
                                 "!");
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World!");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, pieces[0], ", ", pieces[1], " ",
                                 pieces[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, c_strs[0], ", ", c_strs[1], " ",
                                 c_strs[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, "ASCII ", i32s[0], ", ", i32s[1], " ",
                                 i32s[2], "!");
  EXPECT_EQ(result.substr(old_size), "ASCII 72, 67 87!");

  old_size = result.size();
  tensorflow::strings::StrAppend(&result, ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result.substr(old_size), "12345678910, 10987654321!");

  string one = "1";  // Actually, it's the size of this string that we want; a
                     // 64-bit build distinguishes between size_t and uint64,
                     // even though they're both unsigned 64-bit values.
  old_size = result.size();
  tensorflow::strings::StrAppend(&result, "And a ", one.size(), " and a ",
                                 &result[2] - &result[0], " and a ", one,
                                 " 2 3 4", "!");
  EXPECT_EQ(result.substr(old_size), "And a 1 and a 2 and a 1 2 3 4!");

  // result = StrCat("Single chars won't compile", '!');
  // result = StrCat("Neither will NULLs", NULL);
  old_size = result.size();
  tensorflow::strings::StrAppend(
      &result, "To output a char by ASCII/numeric value, use +: ", '!' + 0);
  EXPECT_EQ(result.substr(old_size),
            "To output a char by ASCII/numeric value, use +: 33");

  float f = 100000.5;
  old_size = result.size();
  tensorflow::strings::StrAppend(&result, "A hundred K and a half is ", f);
  EXPECT_EQ(result.substr(old_size), "A hundred K and a half is 100000.5");

  double d = f;
  d *= d;
  old_size = result.size();
  tensorflow::strings::StrAppend(&result, "A hundred K and a half squared is ",
                                 d);
  EXPECT_EQ(result.substr(old_size),
            "A hundred K and a half squared is 10000100000.25");

  // Test 9 arguments, the old maximum
  old_size = result.size();
  tensorflow::strings::StrAppend(&result, 1, 22, 333, 4444, 55555, 666666,
                                 7777777, 88888888, 9);
  EXPECT_EQ(result.substr(old_size), "1223334444555556666667777777888888889");

  // No limit thanks to C++11's variadic templates
  old_size = result.size();
  tensorflow::strings::StrAppend(
      &result, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "a", "b", "c", "d", "e", "f", "g",
      "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
      "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
      "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
      "No limit thanks to C++11's variadic templates");
  EXPECT_EQ(result.substr(old_size),
            "12345678910abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "No limit thanks to C++11's variadic templates");
}

TEST(StrAppend, Death) {
  string s = "self";
  EXPECT_DEBUG_DEATH(strings::StrAppend(&s, s.c_str() + 1), "Check failed:");
  EXPECT_DEBUG_DEATH(strings::StrAppend(&s, s), "Check failed:");
}

static void CheckHex64(uint64 v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc mht_0(mht_0_v, 493, "", "./tensorflow/core/platform/strcat_test.cc", "CheckHex64");

  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::kZeroPad16));
  string expected = Printf("%016llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v, tensorflow::strings::kZeroPad8));
  expected = Printf("%08llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%llx", static_cast<unsigned long long>(v));
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void CheckHex32(uint32 v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc mht_1(mht_1_v, 511, "", "./tensorflow/core/platform/strcat_test.cc", "CheckHex32");

  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::kZeroPad8));
  string expected = Printf("%08x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void CheckHexSigned32(int32_t v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc mht_2(mht_2_v, 525, "", "./tensorflow/core/platform/strcat_test.cc", "CheckHexSigned32");

  using tensorflow::strings::Hex;
  string actual = StrCat(Hex(v, tensorflow::strings::kZeroPad8));
  string expected = Printf("%08x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;

  actual = StrCat(Hex(v));
  expected = Printf("%x", v);
  EXPECT_EQ(expected, actual) << " decimal value " << v;
}

static void TestFastPrints() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcat_testDTcc mht_3(mht_3_v, 539, "", "./tensorflow/core/platform/strcat_test.cc", "TestFastPrints");

  using tensorflow::strings::Hex;

  // Test min int to make sure that works
  for (int i = 0; i < 10000; i++) {
    CheckHex64(i);
    CheckHex32(i);
    CheckHexSigned32(i);
    CheckHexSigned32(-i);
  }
  CheckHex64(0x123456789abcdef0ull);
  CheckHex32(0x12345678);

  int8_t minus_one_8bit = -1;
  EXPECT_EQ("ff", tensorflow::strings::StrCat(Hex(minus_one_8bit)));

  int16_t minus_one_16bit = -1;
  EXPECT_EQ("ffff", tensorflow::strings::StrCat(Hex(minus_one_16bit)));
}

TEST(Numbers, TestFunctionsMovedOverFromNumbersMain) { TestFastPrints(); }

}  // namespace strings
}  // namespace tensorflow
