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
class MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc() {
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

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "absl/meta/type_traits.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace strings {

AlphaNum::AlphaNum(Hex hex) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/platform/strcat.cc", "AlphaNum::AlphaNum");

  char *const end = &digits_[kFastToBufferSize];
  char *writer = end;
  uint64 value = hex.value;
  uint64 width = hex.spec;
  // We accomplish minimum width by OR'ing in 0x10000 to the user's value,
  // where 0x10000 is the smallest hex number that is as wide as the user
  // asked for.
  uint64 mask = (static_cast<uint64>(1) << (width - 1) * 4) | value;
  static const char hexdigits[] = "0123456789abcdef";
  do {
    *--writer = hexdigits[value & 0xF];
    value >>= 4;
    mask >>= 4;
  } while (mask != 0);
  piece_ = StringPiece(writer, end - writer);
}

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or integers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, and integer values.
// ----------------------------------------------------------------------

// Append is merely a version of memcpy that returns the address of the byte
// after the area just overwritten.  It comes in multiple flavors to minimize
// call overhead.
static char *Append1(char *out, const AlphaNum &x) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("out: \"" + (out == nullptr ? std::string("nullptr") : std::string((char*)out)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/platform/strcat.cc", "Append1");

  memcpy(out, x.data(), x.size());
  return out + x.size();
}

static char *Append2(char *out, const AlphaNum &x1, const AlphaNum &x2) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("out: \"" + (out == nullptr ? std::string("nullptr") : std::string((char*)out)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/platform/strcat.cc", "Append2");

  memcpy(out, x1.data(), x1.size());
  out += x1.size();

  memcpy(out, x2.data(), x2.size());
  return out + x2.size();
}

static char *Append4(char *out, const AlphaNum &x1, const AlphaNum &x2,
                     const AlphaNum &x3, const AlphaNum &x4) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("out: \"" + (out == nullptr ? std::string("nullptr") : std::string((char*)out)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/platform/strcat.cc", "Append4");

  memcpy(out, x1.data(), x1.size());
  out += x1.size();

  memcpy(out, x2.data(), x2.size());
  out += x2.size();

  memcpy(out, x3.data(), x3.size());
  out += x3.size();

  memcpy(out, x4.data(), x4.size());
  return out + x4.size();
}

string StrCat(const AlphaNum &a) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/platform/strcat.cc", "StrCat");
 return string(a.data(), a.size()); }

string StrCat(const AlphaNum &a, const AlphaNum &b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/platform/strcat.cc", "StrCat");

  string result(a.size() + b.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/platform/strcat.cc", "StrCat");

  string result(a.size() + b.size() + c.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/platform/strcat.cc", "StrCat");

  string result(a.size() + b.size() + c.size() + d.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append4(begin, a, b, c, d);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

namespace {
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_8(mht_8_v, 317, "", "./tensorflow/core/platform/strcat.cc", "Resize");
 s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_9(mht_9_v, 329, "", "./tensorflow/core/platform/strcat.cc", "Resize");

    s->__resize_default_init(new_size);
  }
};

static inline void STLStringResizeUninitialized(string* s, size_t new_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_10(mht_10_v, 337, "", "./tensorflow/core/platform/strcat.cc", "STLStringResizeUninitialized");

  ResizeUninitializedTraits<string>::Resize(s, new_size);
}

}  // namespace
namespace internal {

// Do not call directly - these are not part of the public API.
string CatPieces(std::initializer_list<StringPiece> pieces) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_11(mht_11_v, 348, "", "./tensorflow/core/platform/strcat.cc", "CatPieces");

  size_t total_size = 0;
  for (const StringPiece piece : pieces) total_size += piece.size();
  string result(total_size, '\0');

  char *const begin = &*result.begin();
  char *out = begin;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result.size());
  return result;
}

// It's possible to call StrAppend with a StringPiece that is itself a fragment
// of the string we're appending to.  However the results of this are random.
// Therefore, check for this in debug mode.  Use unsigned math so we only have
// to do one comparison.
#define DCHECK_NO_OVERLAP(dest, src) \
  DCHECK_GE(uintptr_t((src).data() - (dest).data()), uintptr_t((dest).size()))

void AppendPieces(string *result, std::initializer_list<StringPiece> pieces) {
  size_t old_size = result->size();
  size_t total_size = old_size;
  for (const StringPiece piece : pieces) {
    DCHECK_NO_OVERLAP(*result, piece);
    total_size += piece.size();
  }
  STLStringResizeUninitialized(result, total_size);

  char *const begin = &*result->begin();
  char *out = begin + old_size;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace internal

void StrAppend(string *result, const AlphaNum &a) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_12(mht_12_v, 395, "", "./tensorflow/core/platform/strcat.cc", "StrAppend");

  DCHECK_NO_OVERLAP(*result, a);
  result->append(a.data(), a.size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_13(mht_13_v, 403, "", "./tensorflow/core/platform/strcat.cc", "StrAppend");

  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(result, old_size + a.size() + b.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_14(mht_14_v, 417, "", "./tensorflow/core/platform/strcat.cc", "StrAppend");

  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(result,
                               old_size + a.size() + b.size() + c.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstrcatDTcc mht_15(mht_15_v, 434, "", "./tensorflow/core/platform/strcat.cc", "StrAppend");

  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  DCHECK_NO_OVERLAP(*result, d);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(
      result, old_size + a.size() + b.size() + c.size() + d.size());
  char *const begin = &*result->begin();
  char *out = Append4(begin + old_size, a, b, c, d);
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace strings
}  // namespace tensorflow
