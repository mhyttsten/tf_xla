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

#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh() {
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


#include <functional>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/platform/status.h", "Status");
}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(tensorflow::error::Code code, tensorflow::StringPiece msg)
      : Status(code, msg, {}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_1(mht_1_v, 223, "", "./tensorflow/core/platform/status.h", "Status");
}

  /// \brief Create a status with the specified error code, msg, and stack trace
  /// as a human-readable string containing more detailed information.
#ifndef SWIG
  Status(tensorflow::error::Code code, tensorflow::StringPiece msg,
         std::vector<StackFrame>&& stack_trace);
#endif

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);
#ifndef SWIG
  Status(Status&& s) noexcept;
  Status& operator=(Status&& s) noexcept;
#endif  // SWIG

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/platform/status.h", "ok");
 return (state_ == nullptr); }

  tensorflow::error::Code code() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/platform/status.h", "code");

    return ok() ? tensorflow::error::OK : state_->code;
  }

  const std::string& error_message() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_4(mht_4_v, 258, "", "./tensorflow/core/platform/status.h", "error_message");

    return ok() ? empty_string() : state_->msg;
  }

  const std::vector<StackFrame>& stack_trace() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_5(mht_5_v, 265, "", "./tensorflow/core/platform/status.h", "stack_trace");

    return ok() ? empty_stack_trace() : state_->stack_trace;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  ///
  /// By default, it returns combination of the error code name, the message and
  /// any associated payload messages. This string is designed simply to be
  /// human readable and its exact format should not be load bearing. Do not
  /// depend on the exact format of the result of `ToString()` which is subject
  /// to change.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  //----------------------------------------------------------------------------
  // Payload Management APIs (Cloned from absl::Status)
  //----------------------------------------------------------------------------
  // A payload may be attached to a status to provide additional context to an
  // error that may not be satisfied by an existing `tensorflow::error::Code`.
  // Typically, this payload serves one of several purposes:
  //
  //   * It may provide more fine-grained semantic information about the error
  //     to facilitate actionable remedies.
  //   * It may provide human-readable contexual information that is more
  //     appropriate to display to an end user.
  //
  // A payload consists of a [key,value] pair, where the key is a string
  // referring to a unique "type URL" and the value is an object of type
  // `absl::Cord` to hold the contextual data.
  //
  // The "type URL" should be unique and follow the format of a URL
  // (https://en.wikipedia.org/wiki/URL) and, ideally, provide some
  // documentation or schema on how to interpret its associated data. For
  // example, the default type URL for a protobuf message type is
  // "type.googleapis.com/packagename.messagename". Other custom wire formats
  // should define the format of type URL in a similar practice so as to
  // minimize the chance of conflict between type URLs.
  // Users should ensure that the type URL can be mapped to a concrete
  // C++ type if they want to deserialize the payload and read it effectively.
  //
  // To attach a payload to a status object, call `Status::SetPayload()`,
  // passing it the type URL and an `absl::Cord` of associated data. Similarly,
  // to extract the payload from a status, call `Status::GetPayload()`. You
  // may attach multiple payloads (with differing type URLs) to any given
  // status object, provided that the status is currently exhibiting an error
  // code (i.e. is not OK).
  // TODO(b/197552541): Use absl::Cord for payload value type.

  // The Payload-related APIs are cloned from absl::Status.
  //
  // Returns the payload of a status given its unique `type_url` key, if
  // present.
  absl::optional<tensorflow::StringPiece> GetPayload(
      tensorflow::StringPiece type_url) const;

  // Sets the payload for a non-ok status using a `type_url` key, overwriting
  // any existing payload for that `type_url`.
  //
  // This function does nothing if the Status is ok.
  void SetPayload(tensorflow::StringPiece type_url,
                  tensorflow::StringPiece payload);

  // Erases the payload corresponding to the `type_url` key.  Returns `true` if
  // the payload was present.
  bool ErasePayload(tensorflow::StringPiece type_url);

  // Iterates over the stored payloads and calls the
  // `visitor(type_key, payload)` callable for each one.
  //
  // The order of calls to `visitor()` is not specified and may change at
  // any time and any mutation on the same Status object during visitation is
  // forbidden and could result in undefined behavior.
  void ForEachPayload(
      const std::function<void(tensorflow::StringPiece,
                               tensorflow::StringPiece)>& visitor) const;

 private:
  static const std::string& empty_string();
  static const std::vector<StackFrame>& empty_stack_trace();
  struct State {
    tensorflow::error::Code code;
    std::string msg;
    std::vector<StackFrame> stack_trace;
    std::unordered_map<std::string, std::string> payloads;
  };

  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  StatusGroup();
  // Constructor to form a StatusGroup from any N set of Status arguments.
  // Usage: StatusGroup({status_a, status_b, status_c});
  StatusGroup(std::initializer_list<Status> statuses);

  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static Status MakeDerived(const Status& s);
  static bool IsDerived(const Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.
  static void ConfigureLogHistory();

  // Returns merged payloads of all statuses. In case multiple statuses have the
  // same payload key, non-derived statuses have priority over derived ones,
  // otherwise one payload value will be chosen in an unspecified but
  // deterministic order.
  // NOTE: The payload marking derived statuses as derived will not be returned.
  std::unordered_map<std::string, std::string> GetPayloads() const;

  // Return a merged status with combined child status messages with a summary.
  Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  Status as_concatenated_status() const;

  bool ok() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_6(mht_6_v, 410, "", "./tensorflow/core/platform/status.h", "ok");
 return ok_; }

  // Augment this group with the child status `status`.
  void Update(const Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_7(mht_7_v, 420, "", "./tensorflow/core/platform/status.h", "HasLogMessages");
 return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;

  // Maintain a sorted collection of statuses.
  struct CompareStatus {
    bool operator()(const Status& a, const Status& b) const {
      return a.ToString() > b.ToString();
    }
  };
  // Using std::set instead of absl::btree_set to keep size for certain
  // dependent libraries under the limit.
  std::set<Status, CompareStatus> derived_;
  std::set<Status, CompareStatus> non_derived_;

  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_8(mht_8_v, 444, "", "./tensorflow/core/platform/status.h", "Status::Status");
}

inline Status& Status::operator=(const Status& s) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_9(mht_9_v, 449, "", "./tensorflow/core/platform/status.h", "=");

  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s) noexcept : state_(std::move(s.state_)) {}

inline Status& Status::operator=(Status&& s) noexcept {
  if (state_ != s.state_) {
    state_ = std::move(s.state_);
  }
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern tensorflow::string* TfCheckOpHelperOutOfLine(
    const ::tensorflow::Status& v, const char* msg);

std::string error_name(error::Code code);

inline tensorflow::string* TfCheckOpHelper(::tensorflow::Status v,
                                           const char* msg) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("msg: \"" + (msg == nullptr ? std::string("nullptr") : std::string((char*)msg)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTh mht_10(mht_10_v, 490, "", "./tensorflow/core/platform/status.h", "TfCheckOpHelper");

  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                                \
  while (auto _result = ::tensorflow::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::tensorflow::Status::OK() == (val))) LOG(FATAL)
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_H_
