/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// StatusOr<T> is the union of a Status object and a T object. StatusOr models
// the concept of an object that is either a value, or an error Status
// explaining why such a value is not present. To this end, StatusOr<T> does not
// allow its Status value to be Status::OK.
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example client usage for a StatusOr<T>, where T is not a pointer:
//
//  StatusOr<float> result = DoBigCalculationThatCouldFail();
//  if (result.ok()) {
//    float answer = result.ValueOrDie();
//    printf("Big calculation yielded: %f", answer);
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<T*>:
//
//  StatusOr<Foo*> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo(result.ValueOrDie());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<std::unique_ptr<T>>:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo = std::move(result.ValueOrDie());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example factory implementation returning StatusOr<T*>:
//
//  StatusOr<Foo*> FooFactory::MakeNewFoo(int arg) {
//    if (arg <= 0) {
//      return tensorflow::InvalidArgument("Arg must be positive");
//    } else {
//      return new Foo(arg);
//    }
//  }
//
// Note that the assignment operators require that destroying the currently
// stored value cannot invalidate the argument; in other words, the argument
// cannot be an alias for the current value, or anything owned by the current
// value.
#ifndef TENSORFLOW_CORE_PLATFORM_STATUSOR_H_
#define TENSORFLOW_CORE_PLATFORM_STATUSOR_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh() {
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


#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor_internals.h"

namespace tensorflow {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
template <typename T>
class TF_MUST_USE_RESULT StatusOr;
#endif

template <typename T>
class StatusOr : private internal_statusor::StatusOrData<T>,
                 private internal_statusor::TraitsBase<
                     std::is_copy_constructible<T>::value,
                     std::is_move_constructible<T>::value> {
  template <typename U>
  friend class StatusOr;

  typedef internal_statusor::StatusOrData<T> Base;

 public:
  typedef T element_type;  // DEPRECATED: use `value_type`.
  typedef T value_type;

  // Constructs a new StatusOr with Status::UNKNOWN status.  This is marked
  // 'explicit' to try to catch cases like 'return {};', where people think
  // StatusOr<std::vector<int>> will be initialized with an empty vector,
  // instead of a Status::UNKNOWN status.
  explicit StatusOr();

  // StatusOr<T> will be copy constructible/assignable if T is copy
  // constructible.
  StatusOr(const StatusOr&) = default;
  StatusOr& operator=(const StatusOr&) = default;

  // StatusOr<T> will be move constructible/assignable if T is move
  // constructible.
  StatusOr(StatusOr&&) = default;
  StatusOr& operator=(StatusOr&&) = default;

  // Conversion copy/move constructor, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr(const StatusOr<U>& other);
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr(StatusOr<U>&& other);

  // Conversion copy/move assignment operator, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(const StatusOr<U>& other);
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(StatusOr<U>&& other);

  // Constructs a new StatusOr with the given value. After calling this
  // constructor, calls to ValueOrDie() will succeed, and calls to status() will
  // return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when the return type is StatusOr<T>.
  //
  // REQUIRES: T is copy constructible.
  StatusOr(const T& value);

  // Constructs a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to ValueOrDie() will CHECK-fail.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
  // REQUIRES: !status.ok(). This requirement is DCHECKed.
  // In optimized builds, passing Status::OK() here will have the effect
  // of passing tensorflow::error::INTERNAL as a fallback.
  StatusOr(const Status& status);
  StatusOr& operator=(const Status& status);

  // TODO(b/62186997): Add operator=(T) overloads.

  // Similar to the `const T&` overload.
  //
  // REQUIRES: T is move constructible.
  StatusOr(T&& value);

  // RValue versions of the operations declared above.
  StatusOr(Status&& status);
  StatusOr& operator=(Status&& status);

  // Returns this->status().ok()
  bool ok() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_0(mht_0_v, 334, "", "./tensorflow/core/platform/statusor.h", "ok");
 return this->status_.ok(); }

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK().
  const Status& status() const &;
  Status status() &&;

  // Returns a reference to our current value, or CHECK-fails if !this->ok().
  //
  // Note: for value types that are cheap to copy, prefer simple code:
  //
  //   T value = statusor.ValueOrDie();
  //
  // Otherwise, if the value type is expensive to copy, but can be left
  // in the StatusOr, simply assign to a reference:
  //
  //   T& value = statusor.ValueOrDie();  // or `const T&`
  //
  // Otherwise, if the value type supports an efficient move, it can be
  // used as follows:
  //
  //   T value = std::move(statusor).ValueOrDie();
  //
  // The std::move on statusor instead of on the whole expression enables
  // warnings about possible uses of the statusor object after the move.
  // C++ style guide waiver for ref-qualified overloads granted in cl/143176389
  // See go/ref-qualifiers for more details on such overloads.
  const T& ValueOrDie() const &;
  T& ValueOrDie() &;
  const T&& ValueOrDie() const &&;
  T&& ValueOrDie() &&;

  // Returns a reference to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  //
  // Use this->ok() or `operator bool()` to verify that there is a current
  // value. Alternatively, see ValueOrDie() for a similar API that guarantees
  // CHECK-failing if there is no current value.
  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  // Returns a pointer to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  //
  // Use this->ok() or `operator bool()` to verify that there is a current
  // value.
  const T* operator->() const;
  T* operator->();

  T ConsumeValueOrDie() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_1(mht_1_v, 390, "", "./tensorflow/core/platform/statusor.h", "ConsumeValueOrDie");
 return std::move(ValueOrDie()); }

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

template <typename T>
StatusOr<T>::StatusOr() : Base(Status(tensorflow::error::UNKNOWN, "")) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_2(mht_2_v, 405, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
StatusOr<T>::StatusOr(const T& value) : Base(value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_3(mht_3_v, 411, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
StatusOr<T>::StatusOr(const Status& status) : Base(status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_4(mht_4_v, 417, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(const Status& status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_5(mht_5_v, 423, "", "./tensorflow/core/platform/statusor.h", "=");

  this->Assign(status);
  return *this;
}

template <typename T>
StatusOr<T>::StatusOr(T&& value) : Base(std::move(value)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_6(mht_6_v, 432, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
StatusOr<T>::StatusOr(Status&& status) : Base(std::move(status)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_7(mht_7_v, 438, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(Status&& status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_8(mht_8_v, 444, "", "./tensorflow/core/platform/statusor.h", "=");

  this->Assign(std::move(status));
  return *this;
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>::StatusOr(const StatusOr<U>& other)
    : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_9(mht_9_v, 456, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>& StatusOr<T>::operator=(const StatusOr<U>& other) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_10(mht_10_v, 464, "", "./tensorflow/core/platform/statusor.h", "=");

  if (other.ok())
    this->Assign(other.ValueOrDie());
  else
    this->Assign(other.status());
  return *this;
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>::StatusOr(StatusOr<U>&& other)
    : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_11(mht_11_v, 479, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::StatusOr");
}

template <typename T>
template <typename U,
          typename std::enable_if<std::is_convertible<U, T>::value>::type*>
inline StatusOr<T>& StatusOr<T>::operator=(StatusOr<U>&& other) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_12(mht_12_v, 487, "", "./tensorflow/core/platform/statusor.h", "=");

  if (other.ok()) {
    this->Assign(std::move(other).ValueOrDie());
  } else {
    this->Assign(std::move(other).status());
  }
  return *this;
}

template <typename T>
const Status& StatusOr<T>::status() const & {
  return this->status_;
}
template <typename T>
Status StatusOr<T>::status() && {
  // Note that we copy instead of moving the status here so that
  // ~StatusOrData() can call ok() without invoking UB.
  return ok() ? Status::OK() : this->status_;
}

template <typename T>
const T& StatusOr<T>::ValueOrDie() const & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::ValueOrDie() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::ValueOrDie() const && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::ValueOrDie() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
const T* StatusOr<T>::operator->() const {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
T* StatusOr<T>::operator->() {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
const T& StatusOr<T>::operator*() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::operator*() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::operator*() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::operator*() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
void StatusOr<T>::IgnoreError() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusorDTh mht_13(mht_13_v, 571, "", "./tensorflow/core/platform/statusor.h", "StatusOr<T>::IgnoreError");

  // no-op
}

#define TF_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                             \
  TF_ASSERT_OK_AND_ASSIGN_IMPL(                                         \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr);

#define TF_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)  \
  auto statusor = (rexpr);                                  \
  ASSERT_TRUE(statusor.status().ok()) << statusor.status(); \
  lhs = std::move(statusor.ValueOrDie())

#define TF_STATUS_MACROS_CONCAT_NAME(x, y) TF_STATUS_MACROS_CONCAT_IMPL(x, y)
#define TF_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#define TF_ASSIGN_OR_RETURN(lhs, rexpr) \
  TF_ASSIGN_OR_RETURN_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define TF_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                             \
  if (TF_PREDICT_FALSE(!statusor.ok())) {              \
    return statusor.status();                          \
  }                                                    \
  lhs = std::move(statusor.ValueOrDie())

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUSOR_H_
