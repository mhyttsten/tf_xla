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

#ifndef TENSORFLOW_COMPILER_XLA_STATUS_MACROS_H_
#define TENSORFLOW_COMPILER_XLA_STATUS_MACROS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh() {
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


#include <memory>
#include <ostream>  // NOLINT
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace status_macros {

// This is a useful error message when encountering XLA Compiler errors that
// could be handled with the non-strict AutoJit mode.
extern const char kPossibleAutoJitAlternative[];

// Stream object used to collect error messages in MAKE_ERROR macros
// or append error messages with APPEND_ERROR.  It accepts any
// arguments with operator<< to build an error string, and then has an
// implicit cast operator to Status, which converts the
// logged string to a Status object and returns it, after logging the
// error.  At least one call to operator<< is required; a compile time
// error will be generated if none are given. Errors will only be
// logged by default for certain status codes, as defined in
// IsLoggedByDefault. This class will give ERROR errors if you don't
// retrieve a Status exactly once before destruction.
//
// The class converts into an intermediate wrapper object
// MakeErrorStreamWithOutput to check that the error stream gets at least one
// item of input.
class MakeErrorStream {
 public:
  // Wrapper around MakeErrorStream that only allows for output. This
  // is created as output of the first operator<< call on
  // MakeErrorStream. The bare MakeErrorStream does not have a
  // Status operator. The net effect of that is that you
  // have to call operator<< at least once or else you'll get a
  // compile time error.
  class MakeErrorStreamWithOutput {
   public:
    explicit MakeErrorStreamWithOutput(MakeErrorStream* error_stream)
        : wrapped_error_stream_(error_stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_0(mht_0_v, 230, "", "./tensorflow/compiler/xla/status_macros.h", "MakeErrorStreamWithOutput");
}

    template <typename T>
    MakeErrorStreamWithOutput& operator<<(const T& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_1(mht_1_v, 236, "", "./tensorflow/compiler/xla/status_macros.h", "operator<<");

      *wrapped_error_stream_ << value;
      return *this;
    }

    // Implicit cast operators to Status and StatusOr.
    // Exactly one of these must be called exactly once before destruction.
    operator Status() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/xla/status_macros.h", "Status");
 return wrapped_error_stream_->GetStatus(); }
    template <typename T>
    operator xla::StatusOr<T>() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/status_macros.h", "xla::StatusOr<T>");

      return wrapped_error_stream_->GetStatus();
    }

   private:
    MakeErrorStream* wrapped_error_stream_;

    MakeErrorStreamWithOutput(const MakeErrorStreamWithOutput&) = delete;
    MakeErrorStreamWithOutput& operator=(const MakeErrorStreamWithOutput&) =
        delete;
  };

  // When starting from an existing error status, this determines whether we'll
  // append or prepend to that status's error message.
  enum PriorMessageHandling { kAppendToPriorMessage, kPrependToPriorMessage };

  // Make an error with the given code.
  template <typename ERROR_CODE_TYPE>
  MakeErrorStream(const char* file, int line, ERROR_CODE_TYPE code)
      : impl_(new Impl(file, line, code, this, true)) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_4(mht_4_v, 274, "", "./tensorflow/compiler/xla/status_macros.h", "MakeErrorStream");
}

  template <typename T>
  MakeErrorStreamWithOutput& operator<<(const T& value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_5(mht_5_v, 280, "", "./tensorflow/compiler/xla/status_macros.h", "operator<<");

    CheckNotDone();
    impl_->stream_ << value;
    return impl_->make_error_stream_with_output_wrapper_;
  }

  // When this message is logged (see with_logging()), include the stack trace.
  MakeErrorStream& with_log_stack_trace() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_6(mht_6_v, 290, "", "./tensorflow/compiler/xla/status_macros.h", "with_log_stack_trace");

    impl_->should_log_stack_trace_ = true;
    return *this;
  }

  // Adds RET_CHECK failure text to error message.
  MakeErrorStreamWithOutput& add_ret_check_failure(const char* condition) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("condition: \"" + (condition == nullptr ? std::string("nullptr") : std::string((char*)condition)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_7(mht_7_v, 300, "", "./tensorflow/compiler/xla/status_macros.h", "add_ret_check_failure");

    return *this << "RET_CHECK failure (" << impl_->file_ << ":" << impl_->line_
                 << ") " << condition << " ";
  }

 private:
  class Impl {
   public:
    Impl(const char* file, int line, tensorflow::error::Code code,
         MakeErrorStream* error_stream, bool is_logged_by_default = true);
    Impl(const Status& status, PriorMessageHandling prior_message_handling,
         const char* file, int line, MakeErrorStream* error_stream);

    ~Impl();

    // This must be called exactly once before destruction.
    Status GetStatus();

    void CheckNotDone() const;

   private:
    const char* file_;
    int line_;
    tensorflow::error::Code code_;

    PriorMessageHandling prior_message_handling_ = kAppendToPriorMessage;
    std::string prior_message_;
    bool is_done_;  // true after Status object has been returned
    std::ostringstream stream_;
    bool should_log_;
    int log_severity_;
    bool should_log_stack_trace_;

    // Wrapper around the MakeErrorStream object that has a
    // Status conversion. The first << operator called on
    // MakeErrorStream will return this object, and only this object
    // can implicitly convert to Status. The net effect of
    // this is that you'll get a compile time error if you call
    // MAKE_ERROR etc. without adding any output.
    MakeErrorStreamWithOutput make_error_stream_with_output_wrapper_;

    friend class MakeErrorStream;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
  };

  void CheckNotDone() const;

  // Returns the status. Used by MakeErrorStreamWithOutput.
  Status GetStatus() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_8(mht_8_v, 352, "", "./tensorflow/compiler/xla/status_macros.h", "GetStatus");
 return impl_->GetStatus(); }

  // Store the actual data on the heap to reduce stack frame sizes.
  std::unique_ptr<Impl> impl_;

  MakeErrorStream(const MakeErrorStream&) = delete;
  MakeErrorStream& operator=(const MakeErrorStream&) = delete;
};

// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
 public:
  explicit StatusAdaptorForMacros(Status status) : status_(std::move(status)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_9(mht_9_v, 368, "", "./tensorflow/compiler/xla/status_macros.h", "StatusAdaptorForMacros");
}

  StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
  StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

  explicit operator bool() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_10(mht_10_v, 376, "", "./tensorflow/compiler/xla/status_macros.h", "bool");
 return ABSL_PREDICT_TRUE(status_.ok()); }

  Status&& Consume() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTh mht_11(mht_11_v, 381, "", "./tensorflow/compiler/xla/status_macros.h", "Consume");
 return std::move(status_); }

 private:
  Status status_;
};

}  // namespace status_macros
}  // namespace xla

#define TF_RET_CHECK(condition)                                           \
  while (ABSL_PREDICT_FALSE(!(condition)))                                \
  return xla::status_macros::MakeErrorStream(__FILE__, __LINE__,          \
                                             tensorflow::error::INTERNAL) \
      .with_log_stack_trace()                                             \
      .add_ret_check_failure(#condition)

#endif  // TENSORFLOW_COMPILER_XLA_STATUS_MACROS_H_
