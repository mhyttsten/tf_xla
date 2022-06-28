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

// #status: LEGACY
// #category: Miscellaneous
// #summary: Integral types; prefer util/intops/strong_int.h
// #bugs: Infrastructure > C++ Library Team > util
//
// IntType is a simple template class mechanism for defining "logical"
// integer-like class types that support many of the same functionalities
// as native integer types, but which prevent assignment, construction, and
// other operations from other similar integer-like types.  Essentially, the
// template class IntType<IntTypeName, ValueType> (where ValueType assumes
// valid scalar types such as int, uint, int32, etc) has the additional
// property that it cannot be assigned to or constructed from other IntTypes
// or native integer types of equal or implicitly convertible type.
//
// The class is useful for preventing mingling of integer variables with
// different logical roles or units.  Unfortunately, C++ provides relatively
// good type-safety for user-defined classes but not for integer types.  It is
// essentially up to the user to use nice variable names and comments to prevent
// accidental mismatches, such as confusing a user-index with a group-index or a
// time-in-milliseconds with a time-in-seconds.  The use of typedefs are limited
// in that regard as they do not enforce type-safety.
//
// USAGE -----------------------------------------------------------------------
//
//    DEFINE_INT_TYPE(IntTypeName, ValueType);
//
//  where:
//    IntTypeName: is the desired (unique) name for the "logical" integer type.
//    ValueType: is one of the integral types as defined by base::is_integral
//               (see base/type_traits.h).
//
// DISALLOWED OPERATIONS / TYPE-SAFETY ENFORCEMENT -----------------------------
//
//  Consider these definitions and variable declarations:
//    DEFINE_INT_TYPE(GlobalDocID, int64);
//    DEFINE_INT_TYPE(LocalDocID, int64);
//    GlobalDocID global;
//    LocalDocID local;
//
//  The class IntType prevents:
//
//  1) Assignments of other IntTypes with different IntTypeNames.
//
//    global = local;                  <-- Fails to compile!
//    local = global;                  <-- Fails to compile!
//
//  2) Explicit/implicit conversion from an IntType to another IntType.
//
//    LocalDocID l(global);            <-- Fails to compile!
//    LocalDocID l = global;           <-- Fails to compile!
//
//    void GetGlobalDoc(GlobalDocID global) { }
//    GetGlobalDoc(global);            <-- Compiles fine, types match!
//    GetGlobalDoc(local);             <-- Fails to compile!
//
//  3) Implicit conversion from an IntType to a native integer type.
//
//    void GetGlobalDoc(int64 global) { ...
//    GetGlobalDoc(global);            <-- Fails to compile!
//    GetGlobalDoc(local);             <-- Fails to compile!
//
//    void GetLocalDoc(int32 local) { ...
//    GetLocalDoc(global);             <-- Fails to compile!
//    GetLocalDoc(local);              <-- Fails to compile!
//
//
// SUPPORTED OPERATIONS --------------------------------------------------------
//
// The following operators are supported: unary: ++ (both prefix and postfix),
// +, -, ! (logical not), ~ (one's complement); comparison: ==, !=, <, <=, >,
// >=; numerical: +, -, *, /; assignment: =, +=, -=, /=, *=; stream: <<. Each
// operator allows the same IntTypeName and the ValueType to be used on
// both left- and right-hand sides.
//
// It also supports an accessor value() returning the stored value as ValueType,
// and a templatized accessor value<T>() method that serves as syntactic sugar
// for static_cast<T>(var.value()).  These accessors are useful when assigning
// the stored value into protocol buffer fields and using it as printf args.
//
// The class also defines a hash functor that allows the IntType to be used
// as key to hashable containers such as std::unordered_map and
// std::unordered_set.
//
// We suggest using the IntTypeIndexedContainer wrapper around FixedArray and
// STL vector (see int-type-indexed-container.h) if an IntType is intended to
// be used as an index into these containers.  These wrappers are indexed in a
// type-safe manner using IntTypes to ensure type-safety.
//
// NB: this implementation does not attempt to abide by or enforce dimensional
// analysis on these scalar types.
//
// EXAMPLES --------------------------------------------------------------------
//
//    DEFINE_INT_TYPE(GlobalDocID, int64);
//    GlobalDocID global = 3;
//    cout << global;                      <-- Prints 3 to stdout.
//
//    for (GlobalDocID i(0); i < global; ++i) {
//      cout << i;
//    }                                    <-- Print(ln)s 0 1 2 to stdout
//
//    DEFINE_INT_TYPE(LocalDocID, int64);
//    LocalDocID local;
//    cout << local;                       <-- Prints 0 to stdout it default
//                                             initializes the value to 0.
//
//    local = 5;
//    local *= 2;
//    LocalDocID l(local);
//    cout << l + local;                   <-- Prints 20 to stdout.
//
//    GenericSearchRequest request;
//    request.set_doc_id(global.value());  <-- Uses value() to extract the value
//                                             from the IntType class.
//
// REMARKS ---------------------------------------------------------------------
//
// The following bad usage is permissible although discouraged.  Essentially, it
// involves using the value*() accessors to extract the native integer type out
// of the IntType class.  Keep in mind that the primary reason for the IntType
// class is to prevent *accidental* mingling of similar logical integer types --
// and not type casting from one type to another.
//
//  DEFINE_INT_TYPE(GlobalDocID, int64);
//  DEFINE_INT_TYPE(LocalDocID, int64);
//  GlobalDocID global;
//  LocalDocID local;
//
//  global = local.value();                       <-- Compiles fine.
//
//  void GetGlobalDoc(GlobalDocID global) { ...
//  GetGlobalDoc(local.value());                  <-- Compiles fine.
//
//  void GetGlobalDoc(int64 global) { ...
//  GetGlobalDoc(local.value());                  <-- Compiles fine.

#ifndef TENSORFLOW_LIB_GTL_INT_TYPE_H_
#define TENSORFLOW_LIB_GTL_INT_TYPE_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_typeDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_typeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_typeDTh() {
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


#include <stddef.h>
#include <functional>
#include <iosfwd>
#include <ostream>  // NOLINT
#include <unordered_map>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {

template <typename IntTypeName, typename _ValueType>
class IntType;

// Defines the IntType using value_type and typedefs it to int_type_name.
// The struct int_type_name ## _tag_ trickery is needed to ensure that a new
// type is created per int_type_name.
#define TF_LIB_GTL_DEFINE_INT_TYPE(int_type_name, value_type)          \
  struct int_type_name##_tag_ {};                                      \
  typedef ::tensorflow::gtl::IntType<int_type_name##_tag_, value_type> \
      int_type_name;

// Holds an integer value (of type ValueType) and behaves as a ValueType by
// exposing assignment, unary, comparison, and arithmetic operators.
//
// The template parameter IntTypeName defines the name for the int type and must
// be unique within a binary (the convenient DEFINE_INT_TYPE macro at the end of
// the file generates a unique IntTypeName).  The parameter ValueType defines
// the integer type value (see supported list above).
//
// This class is NOT thread-safe.
template <typename IntTypeName, typename _ValueType>
class IntType {
 public:
  typedef _ValueType ValueType;                      // for non-member operators
  typedef IntType<IntTypeName, ValueType> ThisType;  // Syntactic sugar.

  // Note that this may change from time to time without notice.
  struct Hasher {
    size_t operator()(const IntType& arg) const {
      return static_cast<size_t>(arg.value());
    }
  };

  template <typename H>
  friend H AbslHashValue(H h, const IntType& i) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_typeDTh mht_0(mht_0_v, 370, "", "./tensorflow/core/lib/gtl/int_type.h", "AbslHashValue");

    return H::combine(std::move(h), i.value());
  }

 public:
  // Default c'tor initializing value_ to 0.
  constexpr IntType() : value_(0) {}
  // C'tor explicitly initializing from a ValueType.
  constexpr explicit IntType(ValueType value) : value_(value) {}

  // IntType uses the default copy constructor, destructor and assign operator.
  // The defaults are sufficient and omitting them allows the compiler to add
  // the move constructor/assignment.

  // -- ACCESSORS --------------------------------------------------------------
  // The class provides a value() accessor returning the stored ValueType value_
  // as well as a templatized accessor that is just a syntactic sugar for
  // static_cast<T>(var.value());
  constexpr ValueType value() const { return value_; }

  template <typename ValType>
  constexpr ValType value() const {
    return static_cast<ValType>(value_);
  }

  // -- UNARY OPERATORS --------------------------------------------------------
  ThisType& operator++() {  // prefix ++
    ++value_;
    return *this;
  }
  const ThisType operator++(int v) {  // postfix ++
    ThisType temp(*this);
    ++value_;
    return temp;
  }
  ThisType& operator--() {  // prefix --
    --value_;
    return *this;
  }
  const ThisType operator--(int v) {  // postfix --
    ThisType temp(*this);
    --value_;
    return temp;
  }

  constexpr bool operator!() const { return value_ == 0; }
  constexpr const ThisType operator+() const { return ThisType(value_); }
  constexpr const ThisType operator-() const { return ThisType(-value_); }
  constexpr const ThisType operator~() const { return ThisType(~value_); }

// -- ASSIGNMENT OPERATORS ---------------------------------------------------
// We support the following assignment operators: =, +=, -=, *=, /=, <<=, >>=
// and %= for both ThisType and ValueType.
#define INT_TYPE_ASSIGNMENT_OP(op)                   \
  ThisType& operator op(const ThisType& arg_value) { \
    value_ op arg_value.value();                     \
    return *this;                                    \
  }                                                  \
  ThisType& operator op(ValueType arg_value) {       \
    value_ op arg_value;                             \
    return *this;                                    \
  }
  INT_TYPE_ASSIGNMENT_OP(+=);
  INT_TYPE_ASSIGNMENT_OP(-=);
  INT_TYPE_ASSIGNMENT_OP(*=);
  INT_TYPE_ASSIGNMENT_OP(/=);
  INT_TYPE_ASSIGNMENT_OP(<<=);  // NOLINT
  INT_TYPE_ASSIGNMENT_OP(>>=);  // NOLINT
  INT_TYPE_ASSIGNMENT_OP(%=);
#undef INT_TYPE_ASSIGNMENT_OP

  ThisType& operator=(ValueType arg_value) {
    value_ = arg_value;
    return *this;
  }

 private:
  // The integer value of type ValueType.
  ValueType value_;

  static_assert(std::is_integral<ValueType>::value, "invalid integer type");
} TF_PACKED;

// -- NON-MEMBER STREAM OPERATORS ----------------------------------------------
// We provide the << operator, primarily for logging purposes.  Currently, there
// seems to be no need for an >> operator.
template <typename IntTypeName, typename ValueType>
std::ostream& operator<<(std::ostream& os,  // NOLINT
                         IntType<IntTypeName, ValueType> arg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSint_typeDTh mht_1(mht_1_v, 461, "", "./tensorflow/core/lib/gtl/int_type.h", "operator<<");

  return os << arg.value();
}

// -- NON-MEMBER ARITHMETIC OPERATORS ------------------------------------------
// We support only the +, -, *, and / operators with the same IntType and
// ValueType types.  The reason is to allow simple manipulation on these IDs
// when used as indices in vectors and arrays.
//
// NB: Although it is possible to do IntType * IntType and IntType / IntType,
// it is probably non-sensical from a dimensionality analysis perspective.
#define INT_TYPE_ARITHMETIC_OP(op)                                        \
  template <typename IntTypeName, typename ValueType>                     \
  static inline constexpr IntType<IntTypeName, ValueType> operator op(    \
      IntType<IntTypeName, ValueType> id_1,                               \
      IntType<IntTypeName, ValueType> id_2) {                             \
    return IntType<IntTypeName, ValueType>(id_1.value() op id_2.value()); \
  }                                                                       \
  template <typename IntTypeName, typename ValueType>                     \
  static inline constexpr IntType<IntTypeName, ValueType> operator op(    \
      IntType<IntTypeName, ValueType> id,                                 \
      typename IntType<IntTypeName, ValueType>::ValueType arg_val) {      \
    return IntType<IntTypeName, ValueType>(id.value() op arg_val);        \
  }                                                                       \
  template <typename IntTypeName, typename ValueType>                     \
  static inline constexpr IntType<IntTypeName, ValueType> operator op(    \
      typename IntType<IntTypeName, ValueType>::ValueType arg_val,        \
      IntType<IntTypeName, ValueType> id) {                               \
    return IntType<IntTypeName, ValueType>(arg_val op id.value());        \
  }
INT_TYPE_ARITHMETIC_OP(+);
INT_TYPE_ARITHMETIC_OP(-);
INT_TYPE_ARITHMETIC_OP(*);
INT_TYPE_ARITHMETIC_OP(/);
INT_TYPE_ARITHMETIC_OP(<<);  // NOLINT
INT_TYPE_ARITHMETIC_OP(>>);  // NOLINT
INT_TYPE_ARITHMETIC_OP(%);
#undef INT_TYPE_ARITHMETIC_OP

// -- NON-MEMBER COMPARISON OPERATORS ------------------------------------------
// Static inline comparison operators.  We allow all comparison operators among
// the following types (OP \in [==, !=, <, <=, >, >=]:
//   IntType<IntTypeName, ValueType> OP IntType<IntTypeName, ValueType>
//   IntType<IntTypeName, ValueType> OP ValueType
//   ValueType OP IntType<IntTypeName, ValueType>
#define INT_TYPE_COMPARISON_OP(op)                               \
  template <typename IntTypeName, typename ValueType>            \
  static inline constexpr bool operator op(                      \
      IntType<IntTypeName, ValueType> id_1,                      \
      IntType<IntTypeName, ValueType> id_2) {                    \
    return id_1.value() op id_2.value();                         \
  }                                                              \
  template <typename IntTypeName, typename ValueType>            \
  static inline constexpr bool operator op(                      \
      IntType<IntTypeName, ValueType> id,                        \
      typename IntType<IntTypeName, ValueType>::ValueType val) { \
    return id.value() op val;                                    \
  }                                                              \
  template <typename IntTypeName, typename ValueType>            \
  static inline constexpr bool operator op(                      \
      typename IntType<IntTypeName, ValueType>::ValueType val,   \
      IntType<IntTypeName, ValueType> id) {                      \
    return val op id.value();                                    \
  }
INT_TYPE_COMPARISON_OP(==);  // NOLINT
INT_TYPE_COMPARISON_OP(!=);  // NOLINT
INT_TYPE_COMPARISON_OP(<);   // NOLINT
INT_TYPE_COMPARISON_OP(<=);  // NOLINT
INT_TYPE_COMPARISON_OP(>);   // NOLINT
INT_TYPE_COMPARISON_OP(>=);  // NOLINT
#undef INT_TYPE_COMPARISON_OP

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_INT_TYPE_H_
