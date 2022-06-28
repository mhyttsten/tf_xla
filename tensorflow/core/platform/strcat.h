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

// #status: RECOMMENDED
// #category: operations on strings
// #summary: Merges strings or numbers with no delimiter.
//
#ifndef TENSORFLOW_CORE_PLATFORM_STRCAT_H_
#define TENSORFLOW_CORE_PLATFORM_STRCAT_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh() {
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
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// The AlphaNum type was designed to be used as the parameter type for StrCat().
// Any routine accepting either a string or a number may accept it.
// The basic idea is that by accepting a "const AlphaNum &" as an argument
// to your function, your callers will automatically convert bools, integers,
// and floating point values to strings for you.
//
// NOTE: Use of AlphaNum outside of the "strings" package is unsupported except
// for the specific case of function parameters of type "AlphaNum" or "const
// AlphaNum &". In particular, instantiating AlphaNum directly as a stack
// variable is not supported.
//
// Conversion from 8-bit values is not accepted because if it were, then an
// attempt to pass ':' instead of ":" might result in a 58 ending up in your
// result.
//
// Bools convert to "0" or "1".
//
// Floating point values are converted to a string which, if passed to strtod(),
// would produce the exact same original double (except in case of NaN; all NaNs
// are considered the same value). We try to keep the string short but it's not
// guaranteed to be as short as possible.
//
// You can convert to Hexadecimal output rather than Decimal output using Hex.
// To do this, pass strings::Hex(my_int) as a parameter to StrCat. You may
// specify a minimum field width using a separate parameter, so the equivalent
// of Printf("%04x", my_int) is StrCat(Hex(my_int, strings::kZeroPad4))
//
// This class has implicit constructors.
namespace tensorflow {
namespace strings {

enum PadSpec {
  kNoPad = 1,
  kZeroPad2,
  kZeroPad3,
  kZeroPad4,
  kZeroPad5,
  kZeroPad6,
  kZeroPad7,
  kZeroPad8,
  kZeroPad9,
  kZeroPad10,
  kZeroPad11,
  kZeroPad12,
  kZeroPad13,
  kZeroPad14,
  kZeroPad15,
  kZeroPad16
};

struct Hex {
  uint64 value;
  enum PadSpec spec;
  template <class Int>
  explicit Hex(Int v, PadSpec s = kNoPad) : spec(s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_0(mht_0_v, 253, "", "./tensorflow/core/platform/strcat.h", "Hex");

    // Prevent sign-extension by casting integers to
    // their unsigned counterparts.
    static_assert(
        sizeof(v) == 1 || sizeof(v) == 2 || sizeof(v) == 4 || sizeof(v) == 8,
        "Unknown integer type");
    value = sizeof(v) == 1
                ? static_cast<uint8>(v)
                : sizeof(v) == 2 ? static_cast<uint16>(v)
                                 : sizeof(v) == 4 ? static_cast<uint32>(v)
                                                  : static_cast<uint64>(v);
  }
};

class AlphaNum {
 public:
  // No bool ctor -- bools convert to an integral type.
  // A bool ctor would also convert incoming pointers (bletch).

  AlphaNum(int i32)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt32ToBufferLeft(i32, digits_)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_1(mht_1_v, 276, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(unsigned int u32)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt32ToBufferLeft(u32, digits_)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_2(mht_2_v, 281, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(long x)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt64ToBufferLeft(x, digits_)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_3(mht_3_v, 286, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(unsigned long x)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt64ToBufferLeft(x, digits_)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_4(mht_4_v, 291, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(long long int i64)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt64ToBufferLeft(i64, digits_)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_5(mht_5_v, 296, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(unsigned long long int u64)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt64ToBufferLeft(u64, digits_)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_6(mht_6_v, 301, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}

  AlphaNum(float f)  // NOLINT(runtime/explicit)
      : piece_(digits_, FloatToBuffer(f, digits_)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_7(mht_7_v, 307, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(double f)  // NOLINT(runtime/explicit)
      : piece_(digits_, DoubleToBuffer(f, digits_)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_8(mht_8_v, 312, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(bfloat16 bf)  // NOLINT(runtime/explicit)
      : piece_(digits_, FloatToBuffer(static_cast<float>(bf), digits_)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_9(mht_9_v, 317, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}

  AlphaNum(Hex hex);               // NOLINT(runtime/explicit)

  AlphaNum(const char *c_str) : piece_(c_str) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("c_str: \"" + (c_str == nullptr ? std::string("nullptr") : std::string((char*)c_str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_10(mht_10_v, 325, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}   // NOLINT(runtime/explicit)
  AlphaNum(const StringPiece &pc) : piece_(pc) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_11(mht_11_v, 329, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}  // NOLINT(runtime/explicit)
  AlphaNum(const tensorflow::string &str)          // NOLINT(runtime/explicit)
      : piece_(str) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_12(mht_12_v, 334, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  AlphaNum(const tensorflow::tstring &str)  // NOLINT(runtime/explicit)
      : piece_(str) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_13(mht_13_v, 339, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}
  template <typename A>
  AlphaNum(const std::basic_string<char, std::char_traits<char>, A> &str)
      : piece_(str) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_14(mht_14_v, 345, "", "./tensorflow/core/platform/strcat.h", "AlphaNum");
}  // NOLINT(runtime/explicit)

  StringPiece::size_type size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_15(mht_15_v, 350, "", "./tensorflow/core/platform/strcat.h", "size");
 return piece_.size(); }
  const char *data() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_16(mht_16_v, 354, "", "./tensorflow/core/platform/strcat.h", "data");
 return piece_.data(); }
  StringPiece Piece() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_17(mht_17_v, 358, "", "./tensorflow/core/platform/strcat.h", "Piece");
 return piece_; }

 private:
  StringPiece piece_;
  char digits_[kFastToBufferSize];

  // Use ":" not ':'
  AlphaNum(char c);  // NOLINT(runtime/explicit)

  TF_DISALLOW_COPY_AND_ASSIGN(AlphaNum);
};

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or numbers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, bool values,
//    and numeric values.
//
//    Don't use this for user-visible strings.  The localization process
//    works poorly on strings built up out of fragments.
//
//    For clarity and performance, don't use StrCat when appending to a
//    string.  In particular, avoid using any of these (anti-)patterns:
//      str.append(StrCat(...))
//      str += StrCat(...)
//      str = StrCat(str, ...)
//    where the last is the worse, with the potential to change a loop
//    from a linear time operation with O(1) dynamic allocations into a
//    quadratic time operation with O(n) dynamic allocations.  StrAppend
//    is a better choice than any of the above, subject to the restriction
//    of StrAppend(&str, a, b, c, ...) that none of the a, b, c, ... may
//    be a reference into str.
// ----------------------------------------------------------------------

// For performance reasons, we have specializations for <= 4 args.
std::string StrCat(const AlphaNum &a) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b,
                   const AlphaNum &c) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d) TF_MUST_USE_RESULT;

namespace internal {

// Do not call directly - this is not part of the public API.
std::string CatPieces(std::initializer_list<StringPiece> pieces);
void AppendPieces(std::string *dest, std::initializer_list<StringPiece> pieces);

}  // namespace internal

// Support 5 or more arguments
template <typename... AV>
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d, const AlphaNum &e,
                   const AV &... args) TF_MUST_USE_RESULT;

template <typename... AV>
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d, const AlphaNum &e, const AV &... args) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTh mht_18(mht_18_v, 420, "", "./tensorflow/core/platform/strcat.h", "StrCat");

  return internal::CatPieces({a.Piece(), b.Piece(), c.Piece(), d.Piece(),
                              e.Piece(),
                              static_cast<const AlphaNum &>(args).Piece()...});
}

// ----------------------------------------------------------------------
// StrAppend()
//    Same as above, but adds the output to the given string.
//    WARNING: For speed, StrAppend does not try to check each of its input
//    arguments to be sure that they are not a subset of the string being
//    appended to.  That is, while this will work:
//
//    string s = "foo";
//    s += s;
//
//    This will not (necessarily) work:
//
//    string s = "foo";
//    StrAppend(&s, s);
//
//    Note: while StrCat supports appending up to 26 arguments, StrAppend
//    is currently limited to 9.  That's rarely an issue except when
//    automatically transforming StrCat to StrAppend, and can easily be
//    worked around as consecutive calls to StrAppend are quite efficient.
// ----------------------------------------------------------------------

void StrAppend(std::string *dest, const AlphaNum &a);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d);

// Support 5 or more arguments
template <typename... AV>
inline void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
                      const AlphaNum &c, const AlphaNum &d, const AlphaNum &e,
                      const AV &... args) {
  internal::AppendPieces(dest,
                         {a.Piece(), b.Piece(), c.Piece(), d.Piece(), e.Piece(),
                          static_cast<const AlphaNum &>(args).Piece()...});
}

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STRCAT_H_
