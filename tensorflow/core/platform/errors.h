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

#ifndef TENSORFLOW_CORE_PLATFORM_ERRORS_H_
#define TENSORFLOW_CORE_PLATFORM_ERRORS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh() {
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


#include <sstream>
#include <string>
#include <utility>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace errors {

typedef ::tensorflow::error::Code Code;

namespace internal {

// The DECLARE_ERROR macro below only supports types that can be converted
// into StrCat's AlphaNum. For the other types we rely on a slower path
// through std::stringstream. To add support of a new type, it is enough to
// make sure there is an operator<<() for it:
//
//   std::ostream& operator<<(std::ostream& os, const MyType& foo) {
//     os << foo.ToString();
//     return os;
//   }
// Eventually absl::strings will have native support for this and we will be
// able to completely remove PrepareForStrCat().
template <typename T>
typename std::enable_if<!std::is_convertible<T, strings::AlphaNum>::value,
                        std::string>::type
PrepareForStrCat(const T& t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_0(mht_0_v, 220, "", "./tensorflow/core/platform/errors.h", "PrepareForStrCat");

  std::stringstream ss;
  ss << t;
  return ss.str();
}
inline const strings::AlphaNum& PrepareForStrCat(const strings::AlphaNum& a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_1(mht_1_v, 228, "", "./tensorflow/core/platform/errors.h", "PrepareForStrCat");

  return a;
}

}  // namespace internal

// Maps UNIX errors into a Status.
Status IOError(const string& context, int err_number);

// Returns all payloads from a Status as a key-value map.
inline std::unordered_map<std::string, std::string> GetPayloads(
    const ::tensorflow::Status& status) {
  std::unordered_map<std::string, std::string> payloads;
  status.ForEachPayload(
      [&payloads](tensorflow::StringPiece key, tensorflow::StringPiece value) {
        payloads[std::string(key)] = std::string(value);
      });
  return payloads;
}

// Inserts all given payloads into the given status. Will overwrite existing
// payloads if they exist with the same key.
inline void InsertPayloads(
    ::tensorflow::Status& status,
    const std::unordered_map<std::string, std::string>& payloads) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/platform/errors.h", "InsertPayloads");

  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, payload.second);
  }
}

// Copies all payloads from one Status to another. Will overwrite existing
// payloads in the destination if they exist with the same key.
inline void CopyPayloads(const ::tensorflow::Status& from,
                         ::tensorflow::Status& to) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_3(mht_3_v, 267, "", "./tensorflow/core/platform/errors.h", "CopyPayloads");

  from.ForEachPayload(
      [&to](tensorflow::StringPiece key, tensorflow::StringPiece value) {
        to.SetPayload(key, value);
      });
}

// Creates a new status with the given code, message and payloads.
inline ::tensorflow::Status Create(
    Code code, ::tensorflow::StringPiece message,
    const std::unordered_map<std::string, std::string>& payloads) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_4(mht_4_v, 280, "", "./tensorflow/core/platform/errors.h", "Create");

  Status status(code, message);
  InsertPayloads(status, payloads);
  return status;
}

// Returns a new Status, replacing its message with the given.
inline ::tensorflow::Status CreateWithUpdatedMessage(
    const ::tensorflow::Status& status, ::tensorflow::StringPiece message) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_5(mht_5_v, 291, "", "./tensorflow/core/platform/errors.h", "CreateWithUpdatedMessage");

  return Create(status.code(), message, GetPayloads(status));
}

// Append some context to an error message.  Each time we append
// context put it on a new line, since it is possible for there
// to be several layers of additional context.
template <typename... Args>
void AppendToMessage(::tensorflow::Status* status, Args... args) {
  std::vector<StackFrame> stack_trace = status->stack_trace();
  auto new_status = ::tensorflow::Status(
      status->code(),
      ::tensorflow::strings::StrCat(status->error_message(), "\n\t", args...),
      std::move(stack_trace));
  CopyPayloads(*status, new_status);
  *status = std::move(new_status);
}

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                          \
  do {                                                   \
    ::tensorflow::Status _status = (__VA_ARGS__);        \
    if (TF_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define TF_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)                  \
  do {                                                              \
    ::tensorflow::Status _status = (expr);                          \
    if (TF_PREDICT_FALSE(!_status.ok())) {                          \
      ::tensorflow::errors::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                               \
    }                                                               \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CONST)                                        \
  template <typename... Args>                                             \
  ::tensorflow::Status FUNC(Args... args) {                               \
    return ::tensorflow::Status(                                          \
        ::tensorflow::error::CONST,                                       \
        ::tensorflow::strings::StrCat(                                    \
            ::tensorflow::errors::internal::PrepareForStrCat(args)...));  \
  }                                                                       \
  template <typename... Args>                                             \
  ::tensorflow::Status FUNC##WithPayloads(                                \
      const ::tensorflow::StringPiece& message,                           \
      const std::unordered_map<std::string, std::string>& payloads) {     \
    return errors::Create(::tensorflow::error::CONST, message, payloads); \
  }                                                                       \
  inline bool Is##FUNC(const ::tensorflow::Status& status) {              \
    return status.code() == ::tensorflow::error::CONST;                   \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

#undef DECLARE_ERROR

// Produces a formatted string pattern from the name which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <name>}}
// Note: The pattern below determines the regex _NODEDEF_NAME_RE in the file
// tensorflow/python/client/session.py
// LINT.IfChange
inline std::string FormatNodeNameForError(const std::string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_6(mht_6_v, 378, "", "./tensorflow/core/platform/errors.h", "FormatNodeNameForError");

  return strings::StrCat("{{node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/client/session.py)
template <typename T>
std::string FormatNodeNamesForError(const T& names) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_7(mht_7_v, 386, "", "./tensorflow/core/platform/errors.h", "FormatNodeNamesForError");

  return absl::StrJoin(
      names, ", ", [](std::string* output, const std::string& s) {
        ::tensorflow::strings::StrAppend(output, FormatNodeNameForError(s));
      });
}
// LINT.IfChange
inline std::string FormatColocationNodeForError(const std::string& name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_8(mht_8_v, 397, "", "./tensorflow/core/platform/errors.h", "FormatColocationNodeForError");

  return strings::StrCat("{{colocation_node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/framework/error_interpolation.py)
template <typename T>
std::string FormatColocationNodeForError(const T& names) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_9(mht_9_v, 405, "", "./tensorflow/core/platform/errors.h", "FormatColocationNodeForError");

  return absl::StrJoin(names, ", ",
                       [](std::string* output, const std::string& s) {
                         ::tensorflow::strings::StrAppend(
                             output, FormatColocationNodeForError(s));
                       });
}

inline std::string FormatFunctionForError(const std::string& name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_10(mht_10_v, 417, "", "./tensorflow/core/platform/errors.h", "FormatFunctionForError");

  return strings::StrCat("{{function_node ", name, "}}");
}

inline Status ReplaceErrorFromNonCommunicationOps(const Status s,
                                                  const std::string& op_name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_11(mht_11_v, 426, "", "./tensorflow/core/platform/errors.h", "ReplaceErrorFromNonCommunicationOps");

  assert(IsUnavailable(s));
  return Status(
      error::INTERNAL,
      strings::StrCat(
          s.error_message(), "\nExecuting non-communication op <", op_name,
          "> originally returned UnavailableError, and was replaced by "
          "InternalError to avoid invoking TF network error handling logic."));
}

template <typename T>
std::string FormatOriginalNodeLocationForError(const T& node_names,
                                               const T& func_names) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSerrorsDTh mht_12(mht_12_v, 441, "", "./tensorflow/core/platform/errors.h", "FormatOriginalNodeLocationForError");

  std::vector<std::string> error_message;
  for (int i = 0; i != node_names.size(); ++i) {
    if (i != 0) {
      error_message.push_back(", ");
    }
    if (i < func_names.size()) {
      error_message.push_back(FormatFunctionForError(func_names[i]));
    }
    error_message.push_back(FormatNodeNameForError(node_names[i]));
  }
  return absl::StrJoin(error_message, "");
}

// The CanonicalCode() for non-errors.
using ::tensorflow::error::OK;

}  // namespace errors
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ERRORS_H_
