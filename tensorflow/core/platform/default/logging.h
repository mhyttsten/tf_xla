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

#if defined(_WIN32)
// prevent compile error because MSVC doesn't realize in debug build that
// LOG(FATAL) finally invokes abort()
#pragma warning(disable : 4716)
#endif  // _WIN32

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_LOGGING_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_LOGGING_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh() {
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


// IWYU pragma: private, include "third_party/tensorflow/core/platform/logging.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/logging.h

#include <atomic>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef ERROR

namespace tensorflow {
const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the maximum log level for VLOG statements.
  // E.g., if MaxVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MaxVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MAX_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  template <typename T>
  void operator&(const T&) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_0(mht_0_v, 257, "", "./tensorflow/core/platform/default/logging.h", "&");
}
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) TF_ATTRIBUTE_COLD;
  TF_ATTRIBUTE_NORETURN ~LogMessageFatal() override;
};

// LogMessageNull supports the DVLOG macro by simply dropping any log messages.
class LogMessageNull : public std::basic_ostringstream<char> {
 public:
  LogMessageNull() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_1(mht_1_v, 274, "", "./tensorflow/core/platform/default/logging.h", "LogMessageNull");
}
  ~LogMessageNull() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_2(mht_2_v, 278, "", "./tensorflow/core/platform/default/logging.h", "~LogMessageNull");
}
};

#define _TF_LOG_INFO \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::INFO)
#define _TF_LOG_WARNING \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::WARNING)
#define _TF_LOG_ERROR \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::ERROR)
#define _TF_LOG_FATAL \
  ::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__)

#define _TF_LOG_QFATAL _TF_LOG_FATAL

#define LOG(severity) _TF_LOG_##severity

#ifdef IS_MOBILE_PLATFORM

// Turn VLOG off when under mobile devices for considerations of binary size.
#define VLOG_IS_ON(lvl) ((lvl) <= 0)

#else

// Otherwise, set TF_CPP_MAX_VLOG_LEVEL environment to update minimum log level
// of VLOG, or TF_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                                     \
  (([](int level, const char* fname) {                                      \
    static const bool vmodule_activated =                                   \
        ::tensorflow::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                               \
  })(lvl, __FILE__))

#endif

#define VLOG(level)                                              \
  TF_PREDICT_TRUE(!VLOG_IS_ON(level))                            \
  ? (void)0                                                      \
  : ::tensorflow::internal::Voidifier() &                        \
          ::tensorflow::internal::LogMessage(__FILE__, __LINE__, \
                                             tensorflow::INFO)

// `DVLOG` behaves like `VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef NDEBUG
#define DVLOG VLOG
#else
#define DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::tensorflow::internal::LogMessageNull()
#endif

class LogEveryNState {
 public:
  bool ShouldLog(int n);
  uint32_t counter() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_3(mht_3_v, 335, "", "./tensorflow/core/platform/default/logging.h", "counter");
 return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogFirstNState {
 public:
  bool ShouldLog(int n);
  uint32 counter() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_4(mht_4_v, 347, "", "./tensorflow/core/platform/default/logging.h", "counter");
 return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryPow2State {
 public:
  bool ShouldLog(int ignored);
  uint32 counter() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_5(mht_5_v, 359, "", "./tensorflow/core/platform/default/logging.h", "counter");
 return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryNSecState {
 public:
  bool ShouldLog(double seconds);
  uint32 counter() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_6(mht_6_v, 371, "", "./tensorflow/core/platform/default/logging.h", "counter");
 return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
  // Cycle count according to CycleClock that we should next log at.
  std::atomic<int64_t> next_log_time_cycles_{0};
};

// This macro has a lot going on!
//
// * A local static (`logging_internal_stateful_condition_state`) is
//   declared in a scope such that each `LOG_EVERY_N` (etc.) line has its own
//   state.
// * `COUNTER`, the third variable, is used to support `<< COUNTER`. It is not
//   mangled, so shadowing can be a problem, albeit more of a
//   shoot-yourself-in-the-foot one.  Don't name your variables `COUNTER`.
// * A single for loop can declare state and also test
//   `condition && state.ShouldLog()`, but there's no way to constrain it to run
//   only once (or not at all) without declaring another variable.  The outer
//   for-loop declares this variable (`do_log`).
// * Using for loops instead of if statements means there's no risk of an
//   ambiguous dangling else statement.
#define LOGGING_INTERNAL_STATEFUL_CONDITION(kind, condition, arg)   \
  for (bool logging_internal_stateful_condition_do_log(condition);  \
       logging_internal_stateful_condition_do_log;                  \
       logging_internal_stateful_condition_do_log = false)          \
    for (static ::tensorflow::internal::Log##kind##State            \
             logging_internal_stateful_condition_state;             \
         logging_internal_stateful_condition_do_log &&              \
         logging_internal_stateful_condition_state.ShouldLog(arg);  \
         logging_internal_stateful_condition_do_log = false)        \
      for (const uint32_t COUNTER ABSL_ATTRIBUTE_UNUSED =           \
               logging_internal_stateful_condition_state.counter(); \
           logging_internal_stateful_condition_do_log;              \
           logging_internal_stateful_condition_do_log = false)

// An instance of `LOG_EVERY_N` increments a hidden zero-initialized counter
// every time execution passes through it and logs the specified message when
// the counter's value is a multiple of `n`, doing nothing otherwise.  Each
// instance has its own counter.  The counter's value can be logged by streaming
// the symbol `COUNTER`.  `LOG_EVERY_N` is thread-safe.
// Example:
//
//   for (const auto& user : all_users) {
//     LOG_EVERY_N(INFO, 1000) << "Processing user #" << COUNTER;
//     ProcessUser(user);
//   }
#define LOG_EVERY_N(severity, n)                       \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryN, true, n) \
  LOG(severity)
// `LOG_FIRST_N` behaves like `LOG_EVERY_N` except that the specified message is
// logged when the counter's value is less than `n`.  `LOG_FIRST_N` is
// thread-safe.
#define LOG_FIRST_N(severity, n)                       \
  LOGGING_INTERNAL_STATEFUL_CONDITION(FirstN, true, n) \
  LOG(severity)
// `LOG_EVERY_POW_2` behaves like `LOG_EVERY_N` except that the specified
// message is logged when the counter's value is a power of 2.
// `LOG_EVERY_POW_2` is thread-safe.
#define LOG_EVERY_POW_2(severity)                         \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryPow2, true, 0) \
  LOG(severity)
// An instance of `LOG_EVERY_N_SEC` uses a hidden state variable to log the
// specified message at most once every `n_seconds`.  A hidden counter of
// executions (whether a message is logged or not) is also maintained and can be
// logged by streaming the symbol `COUNTER`.  `LOG_EVERY_N_SEC` is thread-safe.
// Example:
//
//   LOG_EVERY_N_SEC(INFO, 2.5) << "Got " << COUNTER << " cookies so far";
#define LOG_EVERY_N_SEC(severity, n_seconds)                      \
  LOGGING_INTERNAL_STATEFUL_CONDITION(EveryNSec, true, n_seconds) \
  LOG(severity)

// CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define CHECK(condition)              \
  if (TF_PREDICT_FALSE(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// CHECK* macros. It's not encouraged though.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("t: '" + std::string(1, t) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_7(mht_7_v, 463, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline unsigned char GetReferenceableValue(unsigned char t) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("t: '" + std::string(1, t) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_8(mht_8_v, 468, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline signed char GetReferenceableValue(signed char t) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("t: '" + std::string(1, t) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_9(mht_9_v, 473, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline int16 GetReferenceableValue(int16_t t) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_10(mht_10_v, 477, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline uint16 GetReferenceableValue(uint16 t) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_11(mht_11_v, 481, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline int GetReferenceableValue(int t) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_12(mht_12_v, 485, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline unsigned int GetReferenceableValue(unsigned int t) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_13(mht_13_v, 489, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline int64_t GetReferenceableValue(int64_t t) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_14(mht_14_v, 493, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }
inline uint64 GetReferenceableValue(uint64 t) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_15(mht_15_v, 497, "", "./tensorflow/core/platform/default/logging.h", "GetReferenceableValue");
 return t; }

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_16(mht_16_v, 505, "", "./tensorflow/core/platform/default/logging.h", "MakeCheckOpValueString");

  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v);

#if LANG_CXX11
// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v);
#endif

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  explicit CheckOpString(string* str) : str_(str) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_17(mht_17_v, 530, "", "./tensorflow/core/platform/default/logging.h", "CheckOpString");
}
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  explicit operator bool() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_18(mht_18_v, 536, "", "./tensorflow/core/platform/default/logging.h", "bool");
 return TF_PREDICT_FALSE(str_ != nullptr); }
  string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2,
                          const char* exprtext) TF_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
// statement.  See MakeCheckOpString for sample usage.  Other
// approaches were considered: use of a template method (e.g.,
// base::BuildCheckOpString(exprtext, base::Print<T1>, &v1,
// base::Print<T2>, &v2), however this approach has complications
// related to volatile arguments and function-pointer arguments).
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_19(mht_19_v, 561, "", "./tensorflow/core/platform/default/logging.h", "ForVar1");
 return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream* ForVar2();
  // Get the result (inserts the closing ")").
  string* NewString();

 private:
  std::ostringstream* stream_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for CHECK_OP macro.
// We use the full name Check_EQ, Check_NE, etc. in case the file including
// base/logging.h provides its own #defines for the simpler names EQ, NE, etc.
// This happens if, for example, those are used as token names in a
// yacc grammar.
// The (int, int) overload works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
#define TF_DEFINE_CHECK_OP_IMPL(name, op)                                 \
  template <typename T1, typename T2>                                     \
  inline string* name##Impl(const T1& v1, const T2& v2,                   \
                            const char* exprtext) {                       \
    if (TF_PREDICT_TRUE(v1 op v2))                                        \
      return NULL;                                                        \
    else                                                                  \
      return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext); \
  }                                                                       \
  inline string* name##Impl(int v1, int v2, const char* exprtext) {       \
    return name##Impl<int, int>(v1, v2, exprtext);                        \
  }

// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.

TF_DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
// Compilation error with CHECK_EQ(NULL, x)?
// Use CHECK(x == NULL) instead.

inline string* Check_EQImpl(int v1, size_t v2, const char* exprtext) {
  if (TF_PREDICT_FALSE(v1 < 0))
    ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext);

  return Check_EQImpl(size_t(v1), v2, exprtext);
}

inline string* Check_EQImpl(size_t v1, int v2, const char* exprtext) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("exprtext: \"" + (exprtext == nullptr ? std::string("nullptr") : std::string((char*)exprtext)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_20(mht_20_v, 618, "", "./tensorflow/core/platform/default/logging.h", "Check_EQImpl");

  return Check_EQImpl(v2, v1, exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_NE, !=)

inline string* Check_NEImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 < 0) return NULL;

  return Check_NEImpl(size_t(v1), v2, exprtext);
}

inline string* Check_NEImpl(size_t v1, int v2, const char* exprtext) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("exprtext: \"" + (exprtext == nullptr ? std::string("nullptr") : std::string((char*)exprtext)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_21(mht_21_v, 634, "", "./tensorflow/core/platform/default/logging.h", "Check_NEImpl");

  return Check_NEImpl(v2, v1, exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_LE, <=)

inline string* Check_LEImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 <= 0) return NULL;

  return Check_LEImpl(size_t(v1), v2, exprtext);
}

inline string* Check_LEImpl(size_t v1, int v2, const char* exprtext) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("exprtext: \"" + (exprtext == nullptr ? std::string("nullptr") : std::string((char*)exprtext)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_22(mht_22_v, 650, "", "./tensorflow/core/platform/default/logging.h", "Check_LEImpl");

  if (TF_PREDICT_FALSE(v2 < 0))
    return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext);
  return Check_LEImpl(v1, size_t(v2), exprtext);
}

TF_DEFINE_CHECK_OP_IMPL(Check_LT, <)

inline string* Check_LTImpl(int v1, size_t v2, const char* exprtext) {
  if (v1 < 0) return NULL;

  return Check_LTImpl(size_t(v1), v2, exprtext);
}

inline string* Check_LTImpl(size_t v1, int v2, const char* exprtext) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("exprtext: \"" + (exprtext == nullptr ? std::string("nullptr") : std::string((char*)exprtext)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_23(mht_23_v, 668, "", "./tensorflow/core/platform/default/logging.h", "Check_LTImpl");

  if (v2 < 0)
    return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext);
  return Check_LTImpl(v1, size_t(v2), exprtext);
}

// Implement GE,GT in terms of LE,LT
template <typename T1, typename T2>
inline string* Check_GEImpl(const T1& v1, const T2& v2, const char* exprtext) {
  return Check_LEImpl(v2, v1, exprtext);
}

template <typename T1, typename T2>
inline string* Check_GTImpl(const T1& v1, const T2& v2, const char* exprtext) {
  return Check_LTImpl(v2, v1, exprtext);
}

#undef TF_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                     \
  while (::tensorflow::internal::CheckOpString _result{        \
      ::tensorflow::internal::name##Impl(                      \
          ::tensorflow::internal::GetReferenceableValue(val1), \
          ::tensorflow::internal::GetReferenceableValue(val2), \
          #val1 " " #op " " #val2)})                           \
  ::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
#define CHECK_NOTNULL(val)                                 \
  ::tensorflow::internal::CheckNotNull(__FILE__, __LINE__, \
                                       "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// DCHECK_EQ/NE/...
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) \
  while (false && (condition)) LOG(FATAL)

// NDEBUG is defined, so DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _DCHECK_NOP is a helper, and should not be used outside of this file.
#define _TF_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _TF_DCHECK_NOP(x, y)

#endif

// These are for when you don't want a CHECK failure to print a verbose
// stack trace.  The implementation of CHECK* in this file already doesn't.
#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

int64_t MinLogLevelFromEnv();

int64_t MaxVLogLevelFromEnv();

}  // namespace internal

// LogSink support adapted from //base/logging.h
//
// `LogSink` is an interface which can be extended to intercept and process
// all log messages. LogSink implementations must be thread-safe. A single
// instance will be called from whichever thread is performing a logging
// operation.
class TFLogEntry {
  static absl::LogSeverity AsAbslLogSeverity(int severity) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_24(mht_24_v, 775, "", "./tensorflow/core/platform/default/logging.h", "AsAbslLogSeverity");

    return static_cast<absl::LogSeverity>(severity);
  }

 public:
  explicit TFLogEntry(int severity, absl::string_view message)
      : severity_(AsAbslLogSeverity(severity)), message_(message) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("message: \"" + std::string(message.data(), message.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_25(mht_25_v, 785, "", "./tensorflow/core/platform/default/logging.h", "TFLogEntry");
}

  explicit TFLogEntry(int severity, absl::string_view fname, int line,
                      absl::string_view message)
      : severity_(AsAbslLogSeverity(severity)),
        fname_(fname),
        line_(line),
        message_(message) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("fname: \"" + std::string(fname.data(), fname.size()) + "\"");
   mht_26_v.push_back("message: \"" + std::string(message.data(), message.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_26(mht_26_v, 797, "", "./tensorflow/core/platform/default/logging.h", "TFLogEntry");
}

  absl::LogSeverity log_severity() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_27(mht_27_v, 802, "", "./tensorflow/core/platform/default/logging.h", "log_severity");
 return severity_; }
  std::string FName() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_28(mht_28_v, 806, "", "./tensorflow/core/platform/default/logging.h", "FName");
 return fname_; }
  int Line() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_29(mht_29_v, 810, "", "./tensorflow/core/platform/default/logging.h", "Line");
 return line_; }
  std::string ToString() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_30(mht_30_v, 814, "", "./tensorflow/core/platform/default/logging.h", "ToString");
 return message_; }
  absl::string_view text_message() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_31(mht_31_v, 818, "", "./tensorflow/core/platform/default/logging.h", "text_message");
 return message_; }

 private:
  const absl::LogSeverity severity_;
  const std::string fname_;
  int line_ = -1;
  const std::string message_;
};

class TFLogSink {
 public:
  virtual ~TFLogSink() = default;

  // `Send` is called synchronously during the log statement.  The logging
  // module guarantees not to call `Send` concurrently on the same log sink.
  // Implementations should be careful not to call`LOG` or `CHECK` or take
  // any locks that might be held by the `LOG` caller, to avoid deadlock.
  //
  // `e` is guaranteed to remain valid until the subsequent call to
  // `WaitTillSent` completes, so implementations may store a pointer to or
  // copy of `e` (e.g. in a thread local variable) for use in `WaitTillSent`.
  virtual void Send(const TFLogEntry& entry) = 0;

  // `WaitTillSent` blocks the calling thread (the thread that generated a log
  // message) until the sink has finished processing the log message.
  // `WaitTillSent` is called once per log message, following the call to
  // `Send`.  This may be useful when log messages are buffered or processed
  // asynchronously by an expensive log sink.
  // The default implementation returns immediately.  Like `Send`,
  // implementations should be careful not to call `LOG` or `CHECK or take any
  // locks that might be held by the `LOG` caller, to avoid deadlock.
  virtual void WaitTillSent() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSloggingDTh mht_32(mht_32_v, 852, "", "./tensorflow/core/platform/default/logging.h", "WaitTillSent");
}
};

// This is the default log sink. This log sink is used if there are no other
// log sinks registered. To disable the default log sink, set the
// "no_default_logger" Bazel config setting to true or define a
// NO_DEFAULT_LOGGER preprocessor symbol. This log sink will always log to
// stderr.
class TFDefaultLogSink : public TFLogSink {
 public:
  void Send(const TFLogEntry& entry) override;
};

// Add or remove a `LogSink` as a consumer of logging data.  Thread-safe.
void TFAddLogSink(TFLogSink* sink);
void TFRemoveLogSink(TFLogSink* sink);

// Get all the log sinks.  Thread-safe.
std::vector<TFLogSink*> TFGetLogSinks();

// Change verbose level of pre-defined files if envorionment
// variable `env_var` is defined. This is currently a no op.
void UpdateLogVerbosityIfDefined(const char* env_var);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_LOGGING_H_
