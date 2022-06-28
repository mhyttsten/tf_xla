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

#ifndef TENSORFLOW_CORE_PLATFORM_TSTRING_H_
#define TENSORFLOW_CORE_PLATFORM_TSTRING_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPStstringDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPStstringDTh() {
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


#include <assert.h>

#include <ostream>
#include <string>

#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/ctstring.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {

// tensorflow::tstring is the scalar type for DT_STRING tensors.
//
// tstrings are meant to be used when interfacing with string tensors, and
// should not be considered as a general replacement for std::string in
// tensorflow.  The primary purpose of tstring is to provide a unified and
// stable ABI for string tensors across TF Core/C-API/Lite/etc---mitigating
// unnecessary conversions across language boundaries, and allowing for compiler
// agnostic interoperability across dynamically loaded modules.
//
// In addition to ABI stability, tstrings features two string subtypes, VIEW and
// OFFSET.
//
// VIEW tstrings are views into unowned character buffers; they can be used to
// pass around existing character strings without incurring a per object heap
// allocation.  Note that, like std::string_view, it is the user's
// responsibility to ensure that the underlying buffer of a VIEW tstring exceeds
// the lifetime of the associated tstring object.
//
// TODO(dero): Methods for creating OFFSET tensors are not currently
// implemented.
//
// OFFSET tstrings are platform independent offset defined strings which can be
// directly mmaped or copied into a tensor buffer without the need for
// deserialization or processing.  For security reasons, it is imperative that
// OFFSET based string tensors are validated before use, or are from a trusted
// source.
//
// Underlying VIEW and OFFSET buffers are considered immutable, so l-value
// assignment, mutation, or non-const access to data() of tstrings will result
// in the conversion to an owned SMALL/LARGE type.
//
// The interface for tstring largely overlaps with std::string. Except where
// noted, expect equivalent semantics with synonymous std::string methods.
class tstring {
  TF_TString tstr_;

 public:
  enum Type {
    // See cstring.h
    SMALL = TF_TSTR_SMALL,
    LARGE = TF_TSTR_LARGE,
    OFFSET = TF_TSTR_OFFSET,
    VIEW = TF_TSTR_VIEW,
  };

  // Assignment to a tstring object with a tstring::view type will create a VIEW
  // type tstring.
  class view {
    const char* data_;
    size_t size_;

   public:
    explicit view(const char* data, size_t size) : data_(data), size_(size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/platform/tstring.h", "view");
}
    explicit view(const char* data) : data_(data), size_(::strlen(data)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_1(mht_1_v, 257, "", "./tensorflow/core/platform/tstring.h", "view");
}

    const char* data() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_2(mht_2_v, 262, "", "./tensorflow/core/platform/tstring.h", "data");
 return data_; }

    size_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_3(mht_3_v, 267, "", "./tensorflow/core/platform/tstring.h", "size");
 return size_; }

    view() = delete;
    view(const view&) = delete;
    view& operator=(const view&) = delete;
  };

  typedef const char* const_iterator;

  // Ctor
  tstring();
  tstring(const std::string& str);  // NOLINT TODO(b/147740521): Make explicit.
  tstring(const char* str, size_t len);
  tstring(const char* str);  // NOLINT TODO(b/147740521): Make explicit.
  tstring(size_t n, char c);
  explicit tstring(const StringPiece str);
#ifdef PLATFORM_GOOGLE
  explicit tstring(const absl::Cord& cord);
#endif  // PLATFORM_GOOGLE

  // Copy
  tstring(const tstring& str);

  // Move
  tstring(tstring&& str) noexcept;

  // Dtor
  ~tstring();

  // Copy Assignment
  tstring& operator=(const tstring& str);
  tstring& operator=(const std::string& str);
  tstring& operator=(const char* str);
  tstring& operator=(char ch);
  tstring& operator=(const StringPiece str);
#ifdef PLATFORM_GOOGLE
  tstring& operator=(const absl::Cord& cord);
#endif  // PLATFORM_GOOGLE

  // View Assignment
  tstring& operator=(const view& tsv);

  // Move Assignment
  tstring& operator=(tstring&& str);

  // Comparison
  int compare(const char* str, size_t len) const;
  bool operator<(const tstring& o) const;
  bool operator>(const tstring& o) const;
  bool operator==(const char* str) const;
  bool operator==(const tstring& o) const;
  bool operator!=(const char* str) const;
  bool operator!=(const tstring& o) const;

  // Conversion Operators
  // TODO(b/147740521): Make explicit.
  operator std::string() const;  // NOLINT
  // TODO(b/147740521): Make explicit.
  operator StringPiece() const;  // NOLINT
#ifdef PLATFORM_GOOGLE
  template <typename T,
            typename std::enable_if<std::is_same<T, absl::AlphaNum>::value,
                                    T>::type* = nullptr>
  operator T() const;  // NOLINT TODO(b/147740521): Remove.
#endif  // PLATFORM_GOOGLE

  // Attributes
  size_t size() const;
  size_t length() const;
  size_t capacity() const;
  bool empty() const;
  Type type() const;

  // Allocation
  void resize(size_t new_size, char c = 0);
  // Similar to resize, but will leave the newly grown region uninitialized.
  void resize_uninitialized(size_t new_size);
  void clear() noexcept;
  void reserve(size_t n);

  // Iterators
  const_iterator begin() const;
  const_iterator end() const;

  // Const Element Access
  const char* c_str() const;
  const char* data() const;
  const char& operator[](size_t i) const;
  const char& back() const;

  // Mutable Element Access
  // NOTE: For VIEW/OFFSET types, calling these methods will result in the
  // conversion to a SMALL or heap allocated LARGE type.  As a result,
  // previously obtained pointers, references, or iterators to the underlying
  // buffer will point to the original VIEW/OFFSET and not the new allocation.
  char* mdata();
  char* data();  // DEPRECATED: Use mdata().
  char& operator[](size_t i);

  // Assignment
  tstring& assign(const char* str, size_t len);
  tstring& assign(const char* str);

  // View Assignment
  tstring& assign_as_view(const tstring& str);
  tstring& assign_as_view(const std::string& str);
  tstring& assign_as_view(const StringPiece str);
  tstring& assign_as_view(const char* str, size_t len);
  tstring& assign_as_view(const char* str);

  // Modifiers
  // NOTE: Invalid input will result in undefined behavior.
  tstring& append(const tstring& str);
  tstring& append(const char* str, size_t len);
  tstring& append(const char* str);
  tstring& append(size_t n, char c);

  tstring& erase(size_t pos, size_t len);

  tstring& insert(size_t pos, const tstring& str, size_t subpos, size_t sublen);
  tstring& insert(size_t pos, size_t n, char c);
  void swap(tstring& str);
  void push_back(char ch);

  // Friends
  friend bool operator==(const char* a, const tstring& b);
  friend bool operator==(const std::string& a, const tstring& b);
  friend tstring operator+(const tstring& a, const tstring& b);
  friend std::ostream& operator<<(std::ostream& o, const tstring& str);
  friend std::hash<tstring>;
};

// Non-member function overloads

bool operator==(const char* a, const tstring& b);
bool operator==(const std::string& a, const tstring& b);
tstring operator+(const tstring& a, const tstring& b);
std::ostream& operator<<(std::ostream& o, const tstring& str);

// Implementations

// Ctor

inline tstring::tstring() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_4(mht_4_v, 413, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");
 TF_TString_Init(&tstr_); }

inline tstring::tstring(const char* str, size_t len) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_5(mht_5_v, 419, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");

  TF_TString_Init(&tstr_);
  TF_TString_Copy(&tstr_, str, len);
}

inline tstring::tstring(const char* str) : tstring(str, ::strlen(str)) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_6(mht_6_v, 428, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");
}

inline tstring::tstring(size_t n, char c) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_7(mht_7_v, 434, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");

  TF_TString_Init(&tstr_);
  TF_TString_Resize(&tstr_, n, c);
}

inline tstring::tstring(const std::string& str)
    : tstring(str.data(), str.size()) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_8(mht_8_v, 444, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");
}

inline tstring::tstring(const StringPiece str)
    : tstring(str.data(), str.size()) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_9(mht_9_v, 450, "", "./tensorflow/core/platform/tstring.h", "tstring::tstring");
}

#ifdef PLATFORM_GOOGLE
inline tstring::tstring(const absl::Cord& cord) {
  TF_TString_Init(&tstr_);
  TF_TString_ResizeUninitialized(&tstr_, cord.size());

  cord.CopyToArray(data());
}
#endif  // PLATFORM_GOOGLE

// Copy

inline tstring::tstring(const tstring& str) {
  TF_TString_Init(&tstr_);
  TF_TString_Assign(&tstr_, &str.tstr_);
}

// Move

inline tstring::tstring(tstring&& str) noexcept {
  TF_TString_Init(&tstr_);
  TF_TString_Move(&tstr_, &str.tstr_);
}

// Dtor

inline tstring::~tstring() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_10(mht_10_v, 480, "", "./tensorflow/core/platform/tstring.h", "tstring::~tstring");
 TF_TString_Dealloc(&tstr_); }

// Copy Assignment

inline tstring& tstring::operator=(const tstring& str) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_11(mht_11_v, 488, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_Assign(&tstr_, &str.tstr_);

  return *this;
}

inline tstring& tstring::operator=(const std::string& str) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_12(mht_12_v, 498, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_Copy(&tstr_, str.data(), str.size());
  return *this;
}

inline tstring& tstring::operator=(const char* str) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_13(mht_13_v, 507, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_Copy(&tstr_, str, ::strlen(str));

  return *this;
}

inline tstring& tstring::operator=(char c) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_14(mht_14_v, 517, "", "./tensorflow/core/platform/tstring.h", "=");

  resize_uninitialized(1);
  (*this)[0] = c;

  return *this;
}

inline tstring& tstring::operator=(const StringPiece str) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_15(mht_15_v, 527, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_Copy(&tstr_, str.data(), str.size());

  return *this;
}

#ifdef PLATFORM_GOOGLE
inline tstring& tstring::operator=(const absl::Cord& cord) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_16(mht_16_v, 537, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_ResizeUninitialized(&tstr_, cord.size());

  cord.CopyToArray(data());

  return *this;
}
#endif  // PLATFORM_GOOGLE

// View Assignment

inline tstring& tstring::operator=(const tstring::view& tsv) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_17(mht_17_v, 551, "", "./tensorflow/core/platform/tstring.h", "=");

  assign_as_view(tsv.data(), tsv.size());

  return *this;
}

// Move Assignment

inline tstring& tstring::operator=(tstring&& str) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_18(mht_18_v, 562, "", "./tensorflow/core/platform/tstring.h", "=");

  TF_TString_Move(&tstr_, &str.tstr_);

  return *this;
}

// Comparison

inline int tstring::compare(const char* str, size_t len) const {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_19(mht_19_v, 574, "", "./tensorflow/core/platform/tstring.h", "tstring::compare");

  int ret = ::memcmp(data(), str, std::min(len, size()));

  if (ret < 0) return -1;
  if (ret > 0) return +1;

  if (size() < len) return -1;
  if (size() > len) return +1;

  return 0;
}

inline bool tstring::operator<(const tstring& o) const {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("o: \"" + (std::string)o + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_20(mht_20_v, 590, "", "./tensorflow/core/platform/tstring.h", "tstring::operator<");

  return compare(o.data(), o.size()) < 0;
}

inline bool tstring::operator>(const tstring& o) const {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("o: \"" + (std::string)o + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_21(mht_21_v, 598, "", "./tensorflow/core/platform/tstring.h", "tstring::operator>");

  return compare(o.data(), o.size()) > 0;
}

inline bool tstring::operator==(const char* str) const {
  return ::strlen(str) == size() && ::memcmp(data(), str, size()) == 0;
}

inline bool tstring::operator==(const tstring& o) const {
  return o.size() == size() && ::memcmp(data(), o.data(), size()) == 0;
}

inline bool tstring::operator!=(const char* str) const {
  return !(*this == str);
}

inline bool tstring::operator!=(const tstring& o) const {
  return !(*this == o);
}

// Conversion Operators

inline tstring::operator std::string() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_22(mht_22_v, 623, "", "./tensorflow/core/platform/tstring.h", "std::string");

  return std::string(data(), size());
}

inline tstring::operator StringPiece() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_23(mht_23_v, 630, "", "./tensorflow/core/platform/tstring.h", "StringPiece");

  return StringPiece(data(), size());
}

#ifdef PLATFORM_GOOGLE
template <typename T, typename std::enable_if<
                          std::is_same<T, absl::AlphaNum>::value, T>::type*>
inline tstring::operator T() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_24(mht_24_v, 640, "", "./tensorflow/core/platform/tstring.h", "T");

  return T(StringPiece(*this));
}
#endif  // PLATFORM_GOOGLE

// Attributes

inline size_t tstring::size() const { return TF_TString_GetSize(&tstr_); }

inline size_t tstring::length() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_25(mht_25_v, 652, "", "./tensorflow/core/platform/tstring.h", "tstring::length");
 return size(); }

inline size_t tstring::capacity() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_26(mht_26_v, 657, "", "./tensorflow/core/platform/tstring.h", "tstring::capacity");

  return TF_TString_GetCapacity(&tstr_);
}

inline bool tstring::empty() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_27(mht_27_v, 664, "", "./tensorflow/core/platform/tstring.h", "tstring::empty");
 return size() == 0; }

inline tstring::Type tstring::type() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_28(mht_28_v, 669, "", "./tensorflow/core/platform/tstring.h", "tstring::type");

  return static_cast<tstring::Type>(TF_TString_GetType(&tstr_));
}

// Allocation

inline void tstring::resize(size_t new_size, char c) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_29(mht_29_v, 679, "", "./tensorflow/core/platform/tstring.h", "tstring::resize");

  TF_TString_Resize(&tstr_, new_size, c);
}

inline void tstring::resize_uninitialized(size_t new_size) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_30(mht_30_v, 686, "", "./tensorflow/core/platform/tstring.h", "tstring::resize_uninitialized");

  TF_TString_ResizeUninitialized(&tstr_, new_size);
}

inline void tstring::clear() noexcept {
  TF_TString_ResizeUninitialized(&tstr_, 0);
}

inline void tstring::reserve(size_t n) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_31(mht_31_v, 697, "", "./tensorflow/core/platform/tstring.h", "tstring::reserve");
 TF_TString_Reserve(&tstr_, n); }

// Iterators

inline tstring::const_iterator tstring::begin() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_32(mht_32_v, 704, "", "./tensorflow/core/platform/tstring.h", "tstring::begin");
 return &(*this)[0]; }
inline tstring::const_iterator tstring::end() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_33(mht_33_v, 708, "", "./tensorflow/core/platform/tstring.h", "tstring::end");
 return &(*this)[size()]; }

// Element Access

inline const char* tstring::c_str() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_34(mht_34_v, 715, "", "./tensorflow/core/platform/tstring.h", "tstring::c_str");
 return data(); }

inline const char* tstring::data() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_35(mht_35_v, 720, "", "./tensorflow/core/platform/tstring.h", "tstring::data");

  return TF_TString_GetDataPointer(&tstr_);
}

inline const char& tstring::operator[](size_t i) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_36(mht_36_v, 727, "", "./tensorflow/core/platform/tstring.h", "lambda");
 return data()[i]; }

inline const char& tstring::back() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_37(mht_37_v, 732, "", "./tensorflow/core/platform/tstring.h", "tstring::back");
 return (*this)[size() - 1]; }

inline char* tstring::mdata() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_38(mht_38_v, 737, "", "./tensorflow/core/platform/tstring.h", "tstring::mdata");

  return TF_TString_GetMutableDataPointer(&tstr_);
}

inline char* tstring::data() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_39(mht_39_v, 744, "", "./tensorflow/core/platform/tstring.h", "tstring::data");

  // Deprecated
  return mdata();
}

inline char& tstring::operator[](size_t i) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_40(mht_40_v, 752, "", "./tensorflow/core/platform/tstring.h", "lambda");
 return mdata()[i]; }

// Assignment

inline tstring& tstring::assign(const char* str, size_t len) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_41(mht_41_v, 760, "", "./tensorflow/core/platform/tstring.h", "tstring::assign");

  TF_TString_Copy(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::assign(const char* str) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_42(mht_42_v, 770, "", "./tensorflow/core/platform/tstring.h", "tstring::assign");

  assign(str, ::strlen(str));

  return *this;
}

// View Assignment

inline tstring& tstring::assign_as_view(const tstring& str) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_43(mht_43_v, 782, "", "./tensorflow/core/platform/tstring.h", "tstring::assign_as_view");

  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const std::string& str) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_44(mht_44_v, 792, "", "./tensorflow/core/platform/tstring.h", "tstring::assign_as_view");

  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const StringPiece str) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_45(mht_45_v, 801, "", "./tensorflow/core/platform/tstring.h", "tstring::assign_as_view");

  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const char* str, size_t len) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_46(mht_46_v, 811, "", "./tensorflow/core/platform/tstring.h", "tstring::assign_as_view");

  TF_TString_AssignView(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::assign_as_view(const char* str) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_47(mht_47_v, 821, "", "./tensorflow/core/platform/tstring.h", "tstring::assign_as_view");

  assign_as_view(str, ::strlen(str));

  return *this;
}

// Modifiers

inline tstring& tstring::append(const tstring& str) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_48(mht_48_v, 833, "", "./tensorflow/core/platform/tstring.h", "tstring::append");

  TF_TString_Append(&tstr_, &str.tstr_);

  return *this;
}

inline tstring& tstring::append(const char* str, size_t len) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_49(mht_49_v, 843, "", "./tensorflow/core/platform/tstring.h", "tstring::append");

  TF_TString_AppendN(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::append(const char* str) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_50(mht_50_v, 853, "", "./tensorflow/core/platform/tstring.h", "tstring::append");

  append(str, ::strlen(str));

  return *this;
}

inline tstring& tstring::append(size_t n, char c) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_51(mht_51_v, 863, "", "./tensorflow/core/platform/tstring.h", "tstring::append");

  // For append use cases, we want to ensure amortized growth.
  const size_t new_size = size() + n;
  TF_TString_ReserveAmortized(&tstr_, new_size);
  resize(new_size, c);

  return *this;
}

inline tstring& tstring::erase(size_t pos, size_t len) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_52(mht_52_v, 875, "", "./tensorflow/core/platform/tstring.h", "tstring::erase");

  memmove(mdata() + pos, data() + pos + len, size() - len - pos);

  resize(size() - len);

  return *this;
}

inline tstring& tstring::insert(size_t pos, const tstring& str, size_t subpos,
                                size_t sublen) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_53(mht_53_v, 888, "", "./tensorflow/core/platform/tstring.h", "tstring::insert");

  size_t orig_size = size();
  TF_TString_ResizeUninitialized(&tstr_, orig_size + sublen);

  memmove(mdata() + pos + sublen, data() + pos, orig_size - pos);
  memmove(mdata() + pos, str.data() + subpos, sublen);

  return *this;
}

inline tstring& tstring::insert(size_t pos, size_t n, char c) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_54(mht_54_v, 902, "", "./tensorflow/core/platform/tstring.h", "tstring::insert");

  size_t size_ = size();
  TF_TString_ResizeUninitialized(&tstr_, size_ + n);

  memmove(mdata() + pos + n, data() + pos, size_ - pos);
  memset(mdata() + pos, c, n);

  return *this;
}

inline void tstring::swap(tstring& str) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_55(mht_55_v, 915, "", "./tensorflow/core/platform/tstring.h", "tstring::swap");

  // TODO(dero): Invalid for OFFSET (unimplemented).
  std::swap(tstr_, str.tstr_);
}

inline void tstring::push_back(char ch) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("ch: '" + std::string(1, ch) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_56(mht_56_v, 924, "", "./tensorflow/core/platform/tstring.h", "tstring::push_back");
 append(1, ch); }

// Friends

inline bool operator==(const char* a, const tstring& b) {
  return ::strlen(a) == b.size() && ::memcmp(a, b.data(), b.size()) == 0;
}

inline bool operator==(const std::string& a, const tstring& b) {
  return a.size() == b.size() && ::memcmp(a.data(), b.data(), b.size()) == 0;
}

inline tstring operator+(const tstring& a, const tstring& b) {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("a: \"" + (std::string)a + "\"");
   mht_57_v.push_back("b: \"" + (std::string)b + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_57(mht_57_v, 941, "", "./tensorflow/core/platform/tstring.h", "+");

  tstring r;
  r.reserve(a.size() + b.size());
  r.append(a);
  r.append(b);

  return r;
}

inline std::ostream& operator<<(std::ostream& o, const tstring& str) {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStstringDTh mht_58(mht_58_v, 954, "", "./tensorflow/core/platform/tstring.h", "operator<<");

  return o.write(str.data(), str.size());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TSTRING_H_
