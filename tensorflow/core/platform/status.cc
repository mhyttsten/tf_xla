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
class MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc() {
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

#include "tensorflow/core/platform/status.h"

#include <stdio.h>

#include <deque>

#include "absl/base/call_once.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/status.pb.h"

namespace tensorflow {

namespace {

// Log sink is used to collect recent warning and error log messages to be
// attached to the error status.
class StatusLogSink : public TFLogSink {
 public:
  static StatusLogSink* GetInstance() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/platform/status.cc", "GetInstance");

    static StatusLogSink* sink = new StatusLogSink();
    return sink;
  }

  void enable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/platform/status.cc", "enable");

    absl::call_once(flag_, [this] {
      num_messages_ = 5;  // default to 5 messages

      if (const char* num_msgs_str =
              getenv("TF_WORKER_NUM_FORWARDED_LOG_MESSAGES")) {
        if (!absl::SimpleAtoi(num_msgs_str, &num_messages_)) {
          LOG(WARNING) << "Failed to parse env variable "
                          "TF_WORKER_NUM_WARNING_ERROR_LOG_IN_STATUS="
                       << num_msgs_str << " as int. Using the default value "
                       << num_messages_ << ".";
        }
      }

      if (num_messages_ > 0) {
        TFAddLogSink(this);
      }
    });
  }

  void GetMessages(std::vector<std::string>* logs) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);

    for (auto& msg : messages_) {
      logs->push_back(msg);
    }
  }

  void Send(const TFLogEntry& entry) override TF_LOCKS_EXCLUDED(mu_) {
    if (entry.log_severity() < absl::LogSeverity::kWarning) return;

    mutex_lock lock(mu_);
    messages_.emplace_back(entry.ToString());
    if (messages_.size() > static_cast<size_t>(num_messages_)) {
      messages_.pop_front();
    }
  }

 private:
  mutex mu_;
  // for allowing repeated/concurrent calls to enable()
  absl::once_flag flag_;
  int num_messages_ = 0;
  std::deque<std::string> messages_ TF_GUARDED_BY(mu_);
};

}  // namespace

Status::Status(tensorflow::error::Code code, tensorflow::StringPiece msg,
               std::vector<StackFrame>&& stack_trace) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/platform/status.cc", "Status::Status");

  assert(code != tensorflow::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
  state_->stack_trace = std::move(stack_trace);
  VLOG(5) << "Generated non-OK status: \"" << *this << "\". "
          << CurrentStackTrace();
}

void Status::Update(const Status& new_status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_3(mht_3_v, 284, "", "./tensorflow/core/platform/status.cc", "Status::Update");

  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/platform/status.cc", "Status::SlowCopyFrom");

  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string& Status::empty_string() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_5(mht_5_v, 304, "", "./tensorflow/core/platform/status.cc", "Status::empty_string");

  static string* empty = new string;
  return *empty;
}

const std::vector<StackFrame>& Status::empty_stack_trace() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_6(mht_6_v, 312, "", "./tensorflow/core/platform/status.cc", "Status::empty_stack_trace");

  static std::vector<StackFrame>* empty = new std::vector<StackFrame>();
  return *empty;
}

std::string error_name(error::Code code) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_7(mht_7_v, 320, "", "./tensorflow/core/platform/status.cc", "error_name");

  switch (code) {
    case tensorflow::error::OK:
      return "OK";
      break;
    case tensorflow::error::CANCELLED:
      return "CANCELLED";
      break;
    case tensorflow::error::UNKNOWN:
      return "UNKNOWN";
      break;
    case tensorflow::error::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
      break;
    case tensorflow::error::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
      break;
    case tensorflow::error::NOT_FOUND:
      return "NOT_FOUND";
      break;
    case tensorflow::error::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
      break;
    case tensorflow::error::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
      break;
    case tensorflow::error::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
      break;
    case tensorflow::error::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
      break;
    case tensorflow::error::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
      break;
    case tensorflow::error::ABORTED:
      return "ABORTED";
      break;
    case tensorflow::error::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
      break;
    case tensorflow::error::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
      break;
    case tensorflow::error::INTERNAL:
      return "INTERNAL";
      break;
    case tensorflow::error::UNAVAILABLE:
      return "UNAVAILABLE";
      break;
    case tensorflow::error::DATA_LOSS:
      return "DATA_LOSS";
      break;
    default:
      char tmp[30];
      snprintf(tmp, sizeof(tmp), "UNKNOWN_CODE(%d)", static_cast<int>(code));
      return tmp;
      break;
  }
}

std::string Status::ToString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_8(mht_8_v, 384, "", "./tensorflow/core/platform/status.cc", "Status::ToString");

  if (state_ == nullptr) {
    return "OK";
  } else {
    std::string result(error_name(code()));
    result += ": ";
    result += state_->msg;

    for (const std::pair<const std::string, std::string>& element :
         state_->payloads) {
      absl::StrAppend(&result, " [", element.first, "='",
                      absl::CHexEscape(element.second), "']");
    }

    return result;
  }
}

void Status::IgnoreError() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_9(mht_9_v, 405, "", "./tensorflow/core/platform/status.cc", "Status::IgnoreError");

  // no-op
}

void Status::SetPayload(tensorflow::StringPiece type_url,
                        tensorflow::StringPiece payload) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_10(mht_10_v, 413, "", "./tensorflow/core/platform/status.cc", "Status::SetPayload");

  if (ok()) return;
  state_->payloads[std::string(type_url)] = std::string(payload);
}

absl::optional<tensorflow::StringPiece> Status::GetPayload(
    tensorflow::StringPiece type_url) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_11(mht_11_v, 422, "", "./tensorflow/core/platform/status.cc", "Status::GetPayload");

  if (ok()) return absl::nullopt;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return absl::nullopt;
  return tensorflow::StringPiece(payload_iter->second);
}

bool Status::ErasePayload(tensorflow::StringPiece type_url) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_12(mht_12_v, 432, "", "./tensorflow/core/platform/status.cc", "Status::ErasePayload");

  if (ok()) return false;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return false;
  state_->payloads.erase(payload_iter);
  return true;
}

void Status::ForEachPayload(
    const std::function<void(tensorflow::StringPiece, tensorflow::StringPiece)>&
        visitor) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_13(mht_13_v, 445, "", "./tensorflow/core/platform/status.cc", "Status::ForEachPayload");

  if (ok()) return;
  for (const auto& payload : state_->payloads) {
    visitor(payload.first, payload.second);
  }
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_14(mht_14_v, 455, "", "./tensorflow/core/platform/status.cc", "operator<<");

  os << x.ToString();
  return os;
}

std::string* TfCheckOpHelperOutOfLine(const ::tensorflow::Status& v,
                                      const char* msg) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("msg: \"" + (msg == nullptr ? std::string("nullptr") : std::string((char*)msg)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_15(mht_15_v, 465, "", "./tensorflow/core/platform/status.cc", "TfCheckOpHelperOutOfLine");

  std::string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new std::string(r);
}

StatusGroup::StatusGroup() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_16(mht_16_v, 477, "", "./tensorflow/core/platform/status.cc", "StatusGroup::StatusGroup");
}

StatusGroup::StatusGroup(std::initializer_list<Status> statuses) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_17(mht_17_v, 482, "", "./tensorflow/core/platform/status.cc", "StatusGroup::StatusGroup");

  for (const Status& s : statuses) {
    Update(s);
  }
}

static constexpr const char kDerivedStatusProtoUrl[] =
    "type.googleapis.com/tensorflow.DerivedStatus";

Status StatusGroup::MakeDerived(const Status& s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_18(mht_18_v, 494, "", "./tensorflow/core/platform/status.cc", "StatusGroup::MakeDerived");

  if (IsDerived(s)) {
    return s;
  } else {
    Status derived(s);
    // TODO(b/200167936): Serialize an instance of DerivedStatus proto instead
    // of using the string directly. The string is never used so it is not
    // causing any issues at the moment.
    derived.SetPayload(kDerivedStatusProtoUrl, "");
    return derived;
  }
}

bool StatusGroup::IsDerived(const Status& s) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_19(mht_19_v, 510, "", "./tensorflow/core/platform/status.cc", "StatusGroup::IsDerived");

  return s.GetPayload(kDerivedStatusProtoUrl).has_value();
}

void StatusGroup::ConfigureLogHistory() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_20(mht_20_v, 517, "", "./tensorflow/core/platform/status.cc", "StatusGroup::ConfigureLogHistory");

  StatusLogSink::GetInstance()->enable();
}

void StatusGroup::Update(const Status& s) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_21(mht_21_v, 524, "", "./tensorflow/core/platform/status.cc", "StatusGroup::Update");

  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    if (IsDerived(s)) {
      derived_.insert(s);
    } else {
      non_derived_.insert(s);
    }
  }
}

static constexpr int kMaxAggregatedStatusMessageSize = 8 * 1024;
static constexpr int kMaxAttachedLogMessageSize = 512;

std::unordered_map<std::string, std::string> StatusGroup::GetPayloads() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_22(mht_22_v, 543, "", "./tensorflow/core/platform/status.cc", "StatusGroup::GetPayloads");

  std::unordered_map<std::string, std::string> payloads;
  auto capture_payload = [&payloads](tensorflow::StringPiece key,
                                     tensorflow::StringPiece value) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_23(mht_23_v, 549, "", "./tensorflow/core/platform/status.cc", "lambda");

    payloads[std::string(key)] = std::string(value);
  };

  for (const auto& status : derived_) {
    status.ForEachPayload(capture_payload);
  }

  // If a key appears in both derived_ and non_derived_ payloads, then the
  // non_derived_ payload receives priority.
  for (const auto& status : non_derived_) {
    status.ForEachPayload(capture_payload);
  }

  payloads.erase(kDerivedStatusProtoUrl);

  return payloads;
}

Status MakeStatus(
    tensorflow::error::Code code, const tensorflow::StringPiece& message,
    const std::unordered_map<std::string, std::string>& payloads) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_24(mht_24_v, 573, "", "./tensorflow/core/platform/status.cc", "MakeStatus");

  Status status(code, message);
  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, payload.second);
  }
  return status;
}

std::string MakeString(const Status& status) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_25(mht_25_v, 584, "", "./tensorflow/core/platform/status.cc", "MakeString");

  return absl::StrCat(error_name(status.code()), ": ", status.error_message());
}

// Summarize all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are not already summarized.
Status StatusGroup::as_summary_status() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_26(mht_26_v, 593, "", "./tensorflow/core/platform/status.cc", "StatusGroup::as_summary_status");

  if (ok_) {
    return Status::OK();
  }

  // Gather recent logs as a string
  auto get_recent_logs = [this]() -> std::string {
    if (!recent_logs_.empty()) {
      std::vector<std::string> fmt;
      fmt.push_back("\nRecent warning and error logs:");
      for (auto& log : recent_logs_) {
        // Add an indentation to make it look nicer.
        fmt.push_back("  " + log.substr(0, kMaxAttachedLogMessageSize));
      }
      return absl::StrJoin(fmt, "\n");
    } else {
      return "";
    }
  };

  // If only one root status is found, do not add summary header and footer.
  if (non_derived_.size() == 1) {
    return MakeStatus(non_derived_.begin()->code(),
                      strings::StrCat(non_derived_.begin()->error_message(),
                                      get_recent_logs()),
                      GetPayloads());
  }

  if (!non_derived_.empty()) {
    std::vector<std::string> fmt;

    fmt.push_back(
        strings::Printf("%zu root error(s) found.", non_derived_.size()));

    int index = 0;
    auto code = tensorflow::error::CANCELLED;
    for (const auto& s : non_derived_) {
      // NOTE: Avoid using CANCELLED as the code of summary status if the group
      // contains other error code.
      if (code == tensorflow::error::CANCELLED &&
          s.code() != tensorflow::error::CANCELLED) {
        code = s.code();
      }
      fmt.emplace_back(strings::StrCat("  (", index, ") ", MakeString(s)));
      ++index;
    }

    fmt.push_back(strings::Printf("%zu successful operations.", num_ok_));
    fmt.push_back(
        strings::Printf("%zu derived errors ignored.", derived_.size()));

    std::string error_msg =
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize);

    return MakeStatus(code, strings::StrCat(error_msg, get_recent_logs()),
                      GetPayloads());
  } else {
    // All statuses are derived. Pick the first available status to return.
    return MakeDerived(MakeStatus(derived_.begin()->code(),
                                  derived_.begin()->error_message(),
                                  GetPayloads()));
  }
}

// Concatenate all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are already summarized Status.
Status StatusGroup::as_concatenated_status() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_27(mht_27_v, 662, "", "./tensorflow/core/platform/status.cc", "StatusGroup::as_concatenated_status");

  if (ok_) {
    return Status::OK();
  }

  // If only one root status is found, return it directly.
  if (non_derived_.size() == 1) {
    return MakeStatus(non_derived_.begin()->code(),
                      non_derived_.begin()->error_message(), GetPayloads());
  }

  if (!non_derived_.empty()) {
    std::vector<string> fmt;
    fmt.emplace_back("\n=====================");
    for (const auto& s : non_derived_) {
      fmt.emplace_back(MakeString(s));
    }
    fmt.emplace_back("=====================\n");
    return MakeStatus(
        non_derived_.begin()->code(),
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize),
        GetPayloads());
  } else {
    // All statuses are derived. Pick the first available status to return.
    // This should not happen in normal execution.
    return MakeDerived(MakeStatus(derived_.begin()->code(),
                                  derived_.begin()->error_message(),
                                  GetPayloads()));
  }
}

void StatusGroup::AttachLogMessages() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatusDTcc mht_28(mht_28_v, 696, "", "./tensorflow/core/platform/status.cc", "StatusGroup::AttachLogMessages");

  recent_logs_.clear();
  StatusLogSink::GetInstance()->GetMessages(&recent_logs_);
}

}  // namespace tensorflow
