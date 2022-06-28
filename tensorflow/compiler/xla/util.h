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

// Generally useful utility functions that are common to (not specific to any
// given part of) the XLA code base.

#ifndef TENSORFLOW_COMPILER_XLA_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh() {
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


#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// Converts the unsigned integer n into a mixed-radix representation with the
// given bounds (radices). More precisely, if there are K radices, then the
// returned vector digits has K entries and satisfies
//
//   0 <= digits[i] < bounds[i],  for i = 0, ..., K - 1
//
// and FromMixedRadix(digits) == n. The mixed radix representation is unique
// modulo the product of the entries of bounds.
std::vector<int64_t> ToMixedRadix(int64_t n, absl::Span<const int64_t> bounds);

// Logs the provided status message with a backtrace.
//
// For use by Status-factories, logs a backtrace at the point where the status
// is created, such that we can use --vmodule=util=1 to see all status
// creation backtraces.
Status WithLogBacktrace(const Status& status);

// Ranks greater than 8 are very rare, so use InlinedVector<int64_t, 8> to store
// the bounds and indices. And for the rare cases of ranks greater than 8,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
inline constexpr int InlineRank() { return 8; }
using DimensionVector = absl::InlinedVector<int64_t, InlineRank()>;

// RAII timer that logs with a given label the wall clock time duration in human
// readable form. This differs from base's ElapsedTimer primarily in that it
// spits out the human-readable duration form.
//
// Keeps track of global maximum and cumulative times across all invocations.
//
// By default, the timing traces are only printed at VLOG(1) and above:
//
//   XLA_SCOPED_LOGGING_TIMER("fooing bar");  // nop if !VLOG_IS_ON(1).
//
// but you can control this via:
//
//   XLA_SCOPED_LOGGING_TIMER_LEVEL("fooing bar", 2);  // nop if !VLOG_IS_ON(2)
//
#define XLA_SCOPED_LOGGING_TIMER(label) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, 1, __COUNTER__)
#define XLA_SCOPED_LOGGING_TIMER_LEVEL(label, level) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, __COUNTER__)

// Helper for implementing macros above.  Do not use directly.
//
// Forces the evaluation of "counter", which we expect is equal to __COUNTER__.
#define XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, counter) \
  XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter)

// Helper for macros above.  Don't use directly.
#define XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter)      \
  static ::xla::TimerStats XLA_TimerStats##counter;                  \
  ::xla::ScopedLoggingTimer XLA_ScopedLoggingTimerInstance##counter( \
      label, /*enabled=*/VLOG_IS_ON(level), __FILE__, __LINE__,      \
      &XLA_TimerStats##counter);

struct TimerStats {
  absl::Mutex stats_mutex;
  double cumulative_secs ABSL_GUARDED_BY(stats_mutex) = 0;
  double max_secs ABSL_GUARDED_BY(stats_mutex) = 0;
  uint64_t times_called ABSL_GUARDED_BY(stats_mutex) = 0;
};

// RAII timer for XLA_SCOPED_LOGGING_TIMER and XLA_SCOPED_LOGGING_TIMER_LEVEL
// macros above.  Recommended usage is via the macros so you don't have to give
// the timer a name or worry about calling VLOG_IS_ON yourself.
class ScopedLoggingTimer {
 public:
  // label: Label to display for logging.
  // enabled: Whether this timer should do anything at all.
  // file: Filename to display in logging.
  // line: Line number to display in logging.
  // `timer_stats`: unowned non-null pointer which is used to populate the
  // global timer statistics.
  ScopedLoggingTimer(absl::string_view label, bool enabled, const char* file,
                     int line, TimerStats* timer_stats);

  // Stop the timer and log the tracked time. Timer is disabled after this
  // function is called.
  void StopAndLog();

  ~ScopedLoggingTimer();

 private:
  const std::string label_;
  const char* const file_;
  const int line_;
  TimerStats* const timer_stats_;
  uint64_t start_micros_;
  bool enabled_;
};

// Given a vector<T>, returns a Span<char> that points at its
// internals.
//
// Warning: if the vector is updated its storage pointer may change, so use this
// with caution (ideally in limited scopes with temporary lifetimes).
template <typename T>
absl::Span<uint8_t> MutableByteSlice(std::vector<T>* v) {
  return absl::Span<uint8_t>(reinterpret_cast<uint8_t*>(v->data()),
                             v->size() * sizeof(T));
}

// Turns an immutable slice of type T into an immutable slice of bytes with the
// same byte size.
template <typename T>
absl::Span<const uint8_t> CastToByteSlice(absl::Span<const T> slice) {
  return absl::Span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(slice.data()), slice.size() * sizeof(T));
}

// Casts a byte slice to a non-byte type T, checking that the original slice
// length is a multiple of sizeof(T).
template <typename T>
absl::Span<const T> CastByteSlice(absl::Span<const uint8_t> slice) {
  CHECK_EQ(0, slice.size() % sizeof(T));
  return absl::Span<const T>(reinterpret_cast<const T*>(slice.data()),
                             slice.size() / sizeof(T));
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using their
// operator==. Like std::equal, but forces size equality.
template <typename Container1T,
          typename ElementType = typename Container1T::value_type>
bool ContainersEqual(const Container1T& c1,
                     std::initializer_list<ElementType> il) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_0(mht_0_v, 347, "", "./tensorflow/compiler/xla/util.h", "ContainersEqual");

  absl::Span<const ElementType> c2{il};
  return absl::c_equal(c1, c2);
}

#if defined(__cpp_lib_to_underlying) && __cpp_lib_to_underlying >= 202102L
using to_underlying = std::to_underlying;
#else
// Helper function which implements C++23's std::to_underlying.
template <typename T>
constexpr absl::underlying_type_t<T> to_underlying(T value) noexcept {
  return static_cast<absl::underlying_type_t<T>>(value);
}
#endif

// Performs a copy of count values from src to dest, using different strides for
// source and destination. The source starting index is src_base, while the
// destination one is dest_base.
template <typename D, typename S>
void StridedCopy(absl::Span<D> dest, int64_t dest_base, int64_t dest_stride,
                 absl::Span<const S> src, int64_t src_base, int64_t src_stride,
                 int64_t count) {
  for (; count > 0; --count, dest_base += dest_stride, src_base += src_stride) {
    dest[dest_base] = static_cast<D>(src[src_base]);
  }
}

// Adds some context information to the error message in a
// Status.  This is useful as Statuses are
// propagated upwards.
Status AddStatus(Status prior, absl::string_view context);
Status AppendStatus(Status prior, absl::string_view context);

// Status error shorthands -- StrFormat's the arguments to be used as an error
// message and returns a status in the canonical error space.
template <typename... Args>
Status InvalidArgument(const absl::FormatSpec<Args...>& format,
                       const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::InvalidArgument(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status Unimplemented(const absl::FormatSpec<Args...>& format,
                     const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Unimplemented(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status InternalError(const absl::FormatSpec<Args...>& format,
                     const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Internal(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status FailedPrecondition(const absl::FormatSpec<Args...>& format,
                          const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::FailedPrecondition(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status Cancelled(const absl::FormatSpec<Args...>& format, const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Cancelled(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status ResourceExhausted(const absl::FormatSpec<Args...>& format,
                         const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::ResourceExhausted(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status NotFound(const absl::FormatSpec<Args...>& format, const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::NotFound(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status Unavailable(const absl::FormatSpec<Args...>& format,
                   const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Unavailable(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status Unknown(const absl::FormatSpec<Args...>& format, const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Unknown(absl::StrFormat(format, args...)));
}
template <typename... Args>
Status Internal(const absl::FormatSpec<Args...>& format, const Args&... args) {
  return WithLogBacktrace(
      tensorflow::errors::Internal(absl::StrFormat(format, args...)));
}

template <typename... Args>
Status InvalidArgumentStrCat(Args&&... concat) {
  return InvalidArgument("%s", absl::StrCat(std::forward<Args>(concat)...));
}

template <typename... Args>
Status UnimplementedStrCat(Args&&... concat) {
  return Unimplemented("%s", absl::StrCat(std::forward<Args>(concat)...));
}

template <typename... Args>
Status InternalErrorStrCat(Args&&... concat) {
  return InternalError("%s", absl::StrCat(std::forward<Args>(concat)...));
}

template <typename... Args>
Status ResourceExhaustedStrCat(Args&&... concat) {
  return ResourceExhausted("%s", absl::StrCat(std::forward<Args>(concat)...));
}

// Splits the lines of the original, replaces leading whitespace with the prefix
// given by "indentation", and returns the string joined by newlines again. As a
// side effect, any additional trailing whitespace is removed.
//
// Note: even different amounts of leading whitespace on different lines will be
// uniformly replaced with "indentation".
std::string Reindent(absl::string_view original, absl::string_view indentation);

template <typename Container>
int64_t PositionInContainer(const Container& container, int64_t value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_1(mht_1_v, 471, "", "./tensorflow/compiler/xla/util.h", "PositionInContainer");

  return std::distance(container.begin(), absl::c_find(container, value));
}

// Formats the container as a comma-separated string. StrAppend must support
// appending the elements of the container. Prefix is prepended and suffix is
// appended to the returned string.
template <typename Container>
std::string CommaSeparatedString(const Container& c, const char* prefix = "",
                                 const char* suffix = "") {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_2(mht_2_v, 483, "", "./tensorflow/compiler/xla/util.h", "CommaSeparatedString");

  // Not using Join() since the implementation here is simple anyway and this
  // avoids copying the string to append prefix.
  std::string comma_separated = prefix;
  const char* separator = "";
  for (const auto& entry : c) {
    absl::StrAppend(&comma_separated, separator, entry);
    separator = ", ";
  }
  comma_separated += suffix;
  return comma_separated;
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
std::string CommaSeparatedString(const std::initializer_list<T>& c,
                                 const char* prefix = "",
                                 const char* suffix = "") {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_3(mht_3_v, 504, "", "./tensorflow/compiler/xla/util.h", "CommaSeparatedString");

  return CommaSeparatedString<std::initializer_list<T>>(c, prefix, suffix);
}

// Formats the container in the mathematical notation for a vector, e.g. (1, 3,
// 7). StrAppend must support appending the elements of c.
template <typename Container>
std::string VectorString(const Container& c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_4(mht_4_v, 514, "", "./tensorflow/compiler/xla/util.h", "VectorString");

  return CommaSeparatedString(c, "(", ")");
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
std::string VectorString(const std::initializer_list<T>& c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_5(mht_5_v, 524, "", "./tensorflow/compiler/xla/util.h", "VectorString");

  return VectorString<std::initializer_list<T>>(c);
}

// Returns a string which can losslessly round trip to a bfloat.
std::string RoundTripFpToString(tensorflow::bfloat16 value);

// Returns a string which can losslessly round trip to a fp16.
std::string RoundTripFpToString(Eigen::half value);

// Returns a string which can losslessly round trip to a float.
std::string RoundTripFpToString(float value);

// Returns a string which can losslessly round trip to a double.
std::string RoundTripFpToString(double value);

// Returns a PaddingConfig object that represents no padding for the given rank.
PaddingConfig MakeNoPaddingConfig(int64_t rank);

// Returns a PaddingConfig object where 'padding' contains
// (low edge padding, high edge padding) pairs for each dimension.
PaddingConfig MakeEdgePaddingConfig(
    absl::Span<const std::pair<int64_t, int64_t>> padding);

// Returns true if the padding configuration has at least one dimension with
// non-zero interior padding.
bool HasInteriorPadding(const PaddingConfig& config);

// Imports the templated FloorOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
T FloorOfRatio(T dividend, T divisor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_6(mht_6_v, 558, "", "./tensorflow/compiler/xla/util.h", "FloorOfRatio");

  return tensorflow::MathUtil::FloorOfRatio<T>(dividend, divisor);
}

// Imports the templated CeilOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
T CeilOfRatio(T dividend, T divisor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_7(mht_7_v, 568, "", "./tensorflow/compiler/xla/util.h", "CeilOfRatio");

  return tensorflow::MathUtil::CeilOfRatio<T>(dividend, divisor);
}

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpTo(13, 8) => 16
template <typename T>
T RoundUpTo(T value, T divisor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_8(mht_8_v, 578, "", "./tensorflow/compiler/xla/util.h", "RoundUpTo");

  return CeilOfRatio(value, divisor) * divisor;
}

// Rounds the value down to a multiple of the divisor by first calling
// FloorOfRatio then multiplying by the divisor. For example:
// RoundDownTo(13, 8) => 8
template <typename T>
T RoundDownTo(T value, T divisor) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_9(mht_9_v, 589, "", "./tensorflow/compiler/xla/util.h", "RoundDownTo");

  return FloorOfRatio(value, divisor) * divisor;
}

template <typename T>
struct DivMod {
  T quotient;
  T modulo;
};

// Divide `dividend` by `divisor` such that the quotient is rounded towards
// negative infinity. The remainder will have the same sign as `divisor`.
template <typename T>
DivMod<T> FloorDivMod(T dividend, T divisor) {
  DivMod<T> div_mod;
  div_mod.quotient = FloorOfRatio(dividend, divisor);
  div_mod.modulo = dividend - div_mod.quotient * divisor;
  return div_mod;
}

// Given a number of flops executed in an amount of time, produces a string that
// represents the throughput;
// e.g. HumanReadableNumFlops(1e9, 1e9) => 1.00GFLOP/s.
std::string HumanReadableNumFlops(double flops, double nanoseconds);

// Given a number of transcendental ops executed in an amount of time, produces
// a string that represents the throughput;
// e.g. HumanReadableNumTranscendentalOps(1e9, 1e9) => 1.00GTROP/s.
std::string HumanReadableNumTranscendentalOps(double trops, double nanoseconds);

// Split the text into multiple lines and log each line with the given
// severity, filename, and line number.
void LogLines(int sev, absl::string_view text, const char* fname, int lineno);

// Used on a function to trap bad calls: any call that matches the specified
// condition will cause a compile-time error. This macro uses a clang-specific
// "diagnose_if" attribute, as described at
// https://clang.llvm.org/docs/AttributeReference.html#diagnose-if
//
// Example:
//
//   int compute_absolute_value(int c)
//     XLA_DIAGNOSE_ERROR_IF(c >= 0, "'c' is already positive.");
#if ABSL_HAVE_ATTRIBUTE(diagnose_if)
#define XLA_DIAGNOSE_ERROR_IF(...) \
  __attribute__((diagnose_if(__VA_ARGS__, "error")))
#else
#define XLA_DIAGNOSE_ERROR_IF(...)
#endif

constexpr bool IsRuntimeEvaluated() {
#ifdef __cpp_lib_is_constant_evaluated
  return !std::is_constant_evaluated();
#elif ABSL_HAVE_BUILTIN(__builtin_is_constant_evaluated)
  return !__builtin_is_constant_evaluated();
#else
  return false;
#endif
}

// Returns a mask with "width" number of least significant bits set.
template <typename T>
constexpr inline T LsbMask(int width)
    XLA_DIAGNOSE_ERROR_IF(width < 0 || width >= std::numeric_limits<T>::digits,
                          "width must be between [0, sizeof(T)*8)") {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  if (IsRuntimeEvaluated()) {
    DCHECK_GE(width, 0) << "Unsupported width " << width;
    DCHECK_LE(width, std::numeric_limits<T>::digits)
        << "Unsupported width " << width;
  }
  return width == 0
             ? 0
             : static_cast<T>(-1) >> (std::numeric_limits<T>::digits - width);
}

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Floor(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return absl::bit_width(x) - 1;
}

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Ceiling(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return x == 0 ? -1 : absl::bit_width(x - 1);
}

// Returns the value with every bit except the lower 'width' bits set to zero.
template <typename T>
constexpr inline T ClearUpperBits(T value, int width) {
  return value & LsbMask<T>(width);
}

// Returns `base` multiplied by itself `exponent` number of times.
//
// Note: returns 1 when `exponent` is zero.
// Precondition: `exponent` is non-negative.
template <typename T>
constexpr T IPow(T base, int exponent)
    XLA_DIAGNOSE_ERROR_IF(exponent < 0, "exponent must be non-negative") {
  if (IsRuntimeEvaluated()) {
    // A negative `exponent` is indicative of a logic bug for integral `base`.
    // We disallow it for floating-point types for symmetry.
    DCHECK_GE(exponent, 0);
  }
  // We use the right-to-left binary exponentiation algorithm.
  T result{1};
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      result *= base;
    }
    base *= base;
    exponent >>= 1;
  }
  return result;
}

template <size_t>
struct UnsignedIntegerTypeForSize;

template <>
struct UnsignedIntegerTypeForSize<1> {
  using type = uint8_t;
};

template <>
struct UnsignedIntegerTypeForSize<2> {
  using type = uint16_t;
};

template <>
struct UnsignedIntegerTypeForSize<4> {
  using type = uint32_t;
};

template <>
struct UnsignedIntegerTypeForSize<8> {
  using type = uint64_t;
};

template <typename T>
constexpr int NanPayloadBits() {
  // Floating point types with NaNs have payloads.
  if (!std::numeric_limits<T>::has_quiet_NaN) {
    return 0;
  }
  return std::numeric_limits<T>::digits - 1;
}

template <typename T>
constexpr uint64_t QuietNanWithoutPayload() {
  if (const int bits = NanPayloadBits<T>()) {
    return uint64_t{1} << (bits - 1);
  }
  return 0;
}

template <typename T>
constexpr uint64_t NanPayloadBitMask() {
  if (const int bits = NanPayloadBits<T>()) {
    return LsbMask<uint64_t>(bits);
  }
  return 0;
}

template <typename T>
T NanWithSignAndPayload(bool sign, uint64_t nan_payload) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_10(mht_10_v, 764, "", "./tensorflow/compiler/xla/util.h", "NanWithSignAndPayload");

  using RepT = typename UnsignedIntegerTypeForSize<sizeof(T)>::type;
  const T val = std::numeric_limits<T>::quiet_NaN();
  auto rep = absl::bit_cast<RepT>(val);
  rep &= LsbMask<RepT>(std::numeric_limits<RepT>::digits - 1);
  rep |= uint64_t{sign} << (std::numeric_limits<RepT>::digits - 1);
  constexpr int kPayloadBits = NanPayloadBits<T>();
  if (kPayloadBits > 0) {
    // Clear rep's NaN payload.
    rep &= ~NanPayloadBitMask<T>();
    CHECK_NE(nan_payload, 0);
    rep |= nan_payload;
  }
  return absl::bit_cast<T>(rep);
}

// Utility for performing a static_cast<> on a std::unique_ptr<>.
template <typename Derived, typename Base>
std::unique_ptr<Derived> unique_ptr_static_cast(std::unique_ptr<Base> ptr) {
  return std::unique_ptr<Derived>(static_cast<Derived*>(ptr.release()));
}

int64_t Product(absl::Span<const int64_t> xs);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < CommonFactors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `CommonFactors(a, b)[CommonFactors(a, b).size - 1] = (a.size, b.size)`
//
// If input and output are the same, return {(0, 0), {1, 1}, ... {a.size,
// b.size}}, otherwise if the given shapes have non-zero size, returns the
// bounds of the shortest possible such subsequences; else, returns `{(0, 0),
// (a.size, b.size)}`.
absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b);

struct ConvertedDimensionNumbers {
  DimensionVector transformed_from_dimensions;
  DimensionVector untransformed_from_dimensions;
  DimensionVector to_dimensions;
  DimensionVector split_from_dimensions;
  DimensionVector split_from_sizes;
  DimensionVector split_to_dimensions;
};

// Convert and unsorted list of dimensions from one shapes dimension sizes to
// another shapes dimensions sizes.
ConvertedDimensionNumbers ConvertDimensionNumbers(
    absl::Span<const int64_t> from_dimensions,
    absl::Span<const int64_t> from_sizes, absl::Span<const int64_t> to_sizes);

// Removes illegal characters from filenames.
std::string SanitizeFileName(std::string file_name);

template <typename C, typename Value>
int64_t FindIndex(const C& c, Value&& value) {
  auto it = absl::c_find(c, std::forward<Value>(value));
  return std::distance(c.begin(), it);
}

template <typename C, typename Value>
void InsertAt(C* c, int64_t index, Value&& value) {
  c->insert(c->begin() + index, std::forward<Value>(value));
}

template <typename C>
void EraseAt(C* c, int64_t index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_11(mht_11_v, 837, "", "./tensorflow/compiler/xla/util.h", "EraseAt");

  c->erase(c->begin() + index);
}

template <typename T>
std::vector<T> SpanToVector(absl::Span<const T> slice) {
  return std::vector<T>(slice.begin(), slice.end());
}

template <typename T, size_t N>
std::vector<T> InlinedVectorToVector(
    const absl::InlinedVector<T, N>& inlined_vector) {
  return std::vector<T>(inlined_vector.begin(), inlined_vector.end());
}

// Returns true if `x` fits in 32-bits.
template <typename T>
bool IsInt32(T x) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_12(mht_12_v, 857, "", "./tensorflow/compiler/xla/util.h", "IsInt32");

  // Following conversion rules: "the value is unchanged if it can be
  // represented in the destination type (and bit-field width); otherwise, the
  // value is implementation-defined."
  return static_cast<int32_t>(x) == x;
}

template <typename T>
Status EraseElementFromVector(std::vector<T>* container, const T& value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSutilDTh mht_13(mht_13_v, 868, "", "./tensorflow/compiler/xla/util.h", "EraseElementFromVector");

  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return Status::OK();
}

// Utility function which splits a double-precision float (F64) into a pair of
// single-precision floating point numbers. The most significant 49 bits (out of
// the total 53 available) in the mantissa of the F64 is represented as the
// unevaluated sum of two non-overlapping single-precision F32s; the 'high' part
// contains 24 bits in its mantissa, and the 'low' part contains 25 bits in its
// sign bit and its mantissa.
// Note: The resulting representation can still only represent 8-bit exponent
// range that is available in F32s (out of a total of 11 exponent bits in F64s).
std::pair<float, float> SplitF64ToF32(double x);

}  // namespace xla

#define XLA_LOG_LINES(SEV, STRING) \
  ::xla::LogLines(SEV, STRING, __FILE__, __LINE__)

#define XLA_VLOG_LINES(LEVEL, STRING)                                 \
  do {                                                                \
    if (VLOG_IS_ON(LEVEL)) XLA_LOG_LINES(::tensorflow::INFO, STRING); \
  } while (false);

// Utility macro that performs the equivalent of what one would expect
// LOG_LINES(FATAL, X) to do but can be used at the end of a function that
// returns a value without getting a compiler warning that no value is returned.
#define XLA_FATAL_LOG(X)                 \
  XLA_LOG_LINES(::tensorflow::ERROR, X); \
  LOG(FATAL) << "Aborting in " << __FUNCTION__ << " due to previous errors.";

#endif  // TENSORFLOW_COMPILER_XLA_UTIL_H_
