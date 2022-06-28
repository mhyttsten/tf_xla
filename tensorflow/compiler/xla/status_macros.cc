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
class MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc() {
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

#include "tensorflow/compiler/xla/status_macros.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {
namespace status_macros {

ABSL_CONST_INIT const char kPossibleAutoJitAlternative[] =
    "This error might be occurring with the use of xla.compile. If it is not "
    "necessary that every Op be compiled with XLA, an alternative is to use "
    "auto_jit with OptimizerOptions.global_jit_level = ON_2 or the environment "
    "variable TF_XLA_FLAGS=\"tf_xla_auto_jit=2\" which will attempt to use xla "
    "to compile as much of the graph as the compiler is able to.";

static Status MakeStatus(tensorflow::error::Code code,
                         const std::string& message) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeStatus");

  return Status(code, message);
}

// Log the error at the given severity, optionally with a stack trace.
// If log_severity is NUM_SEVERITIES, nothing is logged.
static void LogError(const Status& status, const char* filename, int line,
                     int log_severity, bool should_log_stack_trace) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/status_macros.cc", "LogError");

  if (ABSL_PREDICT_TRUE(log_severity != tensorflow::NUM_SEVERITIES)) {
    std::string stack_trace;
    if (should_log_stack_trace) {
      stack_trace = absl::StrCat("\n", tensorflow::CurrentStackTrace());
    }
    switch (log_severity) {
      case tensorflow::INFO:
        LOG(INFO) << status << stack_trace;
        break;
      case tensorflow::WARNING:
        LOG(WARNING) << status << stack_trace;
        break;
      case tensorflow::ERROR:
        LOG(ERROR) << status << stack_trace;
        break;
      case tensorflow::FATAL:
        LOG(FATAL) << status << stack_trace;
        break;
      case tensorflow::NUM_SEVERITIES:
        break;
      default:
        LOG(FATAL) << "Unknown LOG severity " << log_severity;
    }
  }
}

// Make a Status with a code, error message and payload,
// and also send it to LOG(<log_severity>) using the given filename
// and line (unless should_log is false, or log_severity is
// NUM_SEVERITIES).  If should_log_stack_trace is true, the stack
// trace is included in the log message (ignored if should_log is
// false).
static Status MakeError(const char* filename, int line,
                        tensorflow::error::Code code,
                        const std::string& message, bool should_log,
                        int log_severity, bool should_log_stack_trace) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   mht_2_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeError");

  if (ABSL_PREDICT_FALSE(code == tensorflow::error::OK)) {
    LOG(ERROR) << "Cannot create error with status OK";
    code = tensorflow::error::UNKNOWN;
  }
  const Status status = MakeStatus(code, message);
  if (ABSL_PREDICT_TRUE(should_log)) {
    LogError(status, filename, line, log_severity, should_log_stack_trace);
  }
  return status;
}

// This method is written out-of-line rather than in the header to avoid
// generating a lot of inline code for error cases in all callers.
void MakeErrorStream::CheckNotDone() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_3(mht_3_v, 275, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::CheckNotDone");
 impl_->CheckNotDone(); }

MakeErrorStream::Impl::Impl(const char* file, int line,
                            tensorflow::error::Code code,
                            MakeErrorStream* error_stream,
                            bool is_logged_by_default)
    : file_(file),
      line_(line),
      code_(code),
      is_done_(false),
      should_log_(is_logged_by_default),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_4(mht_4_v, 292, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::Impl::Impl");
}

MakeErrorStream::Impl::Impl(const Status& status,
                            PriorMessageHandling prior_message_handling,
                            const char* file, int line,
                            MakeErrorStream* error_stream)
    : file_(file),
      line_(line),
      // Make sure we show some error, even if the call is incorrect.
      code_(!status.ok() ? status.code() : tensorflow::error::UNKNOWN),
      prior_message_handling_(prior_message_handling),
      prior_message_(status.error_message()),
      is_done_(false),
      // Error code type is not visible here, so we can't call
      // IsLoggedByDefault.
      should_log_(true),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_5(mht_5_v, 314, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::Impl::Impl");

  DCHECK(!status.ok()) << "Attempted to append/prepend error text to status OK";
}

MakeErrorStream::Impl::~Impl() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_6(mht_6_v, 321, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::Impl::~Impl");

  // Note: error messages refer to the public MakeErrorStream class.

  if (!is_done_) {
    LOG(ERROR) << "MakeErrorStream destructed without getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

Status MakeErrorStream::Impl::GetStatus() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_7(mht_7_v, 333, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::Impl::GetStatus");

  // Note: error messages refer to the public MakeErrorStream class.

  // Getting a Status object out more than once is not harmful, but
  // it doesn't match the expected pattern, where the stream is constructed
  // as a temporary, loaded with a message, and then casted to Status.
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream got Status more than once: " << file_ << ":"
               << line_ << " " << stream_.str();
  }

  is_done_ = true;

  const std::string& stream_str = stream_.str();
  const std::string str = prior_message_handling_ == kAppendToPriorMessage
                              ? absl::StrCat(prior_message_, stream_str)
                              : absl::StrCat(stream_str, prior_message_);
  if (ABSL_PREDICT_FALSE(str.empty())) {
    return MakeError(
        file_, line_, code_,
        absl::StrCat(str, "Error without message at ", file_, ":", line_),
        true /* should_log */, tensorflow::ERROR /* log_severity */,
        should_log_stack_trace_);
  } else {
    return MakeError(file_, line_, code_, str, should_log_, log_severity_,
                     should_log_stack_trace_);
  }
}

void MakeErrorStream::Impl::CheckNotDone() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macrosDTcc mht_8(mht_8_v, 365, "", "./tensorflow/compiler/xla/status_macros.cc", "MakeErrorStream::Impl::CheckNotDone");

  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream shift called after getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

}  // namespace status_macros
}  // namespace xla
