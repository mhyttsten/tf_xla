/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh() {
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


#include <ostream>
#include <string>
#include <utility>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

// Defines the following utilities:
//
// ===============
// IsOkAndHolds(m)
// ===============
//
// This matcher matches a StatusOr<T> value whose status is OK and whose inner
// value matches matcher m. Example:
//
//   using ::tensorflow::testing::IsOkAndHolds;
//   using ::testing::HasSubstr;
//   ...
//   StatusOr<std::string> status_or_message("Hello, world");
//   EXPECT_THAT(status_or_message, IsOkAndHolds("Hello, world")));
//   EXPECT_THAT(status_or_message, IsOkAndHolds(HasSubstr("Hello,")));
//
// ===============================
// StatusIs(status_code_matcher,
//          error_message_matcher)
// ===============================
//
// This matcher matches a Status or StatusOr<T> if the following are true:
//
//   - the status's code() matches status_code_matcher, and
//   - the status's error_message() matches error_message_matcher.
//
// Example:
//
//   using ::tensorflow::testing::StatusIs;
//   using ::testing::HasSubstr;
//   using ::testing::MatchesRegex;
//   using ::testing::Ne;
//   using ::testing::_;
//   StatusOr<std::string> GetMessage(int id);
//   ...
//
//   // The status code must be CANCELLED; the error message can be anything.
//   EXPECT_THAT(GetName(42),
//               StatusIs(tensorflow::error::CANCELLED, _));
//
//   // The status code can be anything; the error message must match the regex.
//   EXPECT_THAT(GetName(43),
//               StatusIs(_, MatchesRegex("server.*time-out")));
//
//   // The status code should not be CANCELLED; the error message can be
//   // anything with "Cancelled" in it.
//   EXPECT_THAT(GetName(44),
//               StatusIs(Ne(tensorflow::error::CANCELLED),
//                        HasSubstr("Cancelled"))));
//
// =============================
// StatusIs(status_code_matcher)
// =============================
//
// This is a shorthand for
//   StatusIs(status_code_matcher, ::testing::_)
//
// In other words, it's like the two-argument StatusIs(), except that it ignores
// error messages.
//
// ======
// IsOk()
// ======
//
// Matches a Status or StatusOr<T> whose status value is OK.
// Equivalent to 'StatusIs(error::OK)'.
//
// Example:
//   ...
//   StatusOr<std::string> message("Hello, world");
//   EXPECT_THAT(message, IsOk());
//   Status status = Status::OK();
//   EXPECT_THAT(status, IsOk());

namespace tensorflow {

template <typename T>
void PrintTo(const StatusOr<T>& status_or, std::ostream* os) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_0(mht_0_v, 273, "", "./tensorflow/core/platform/status_matchers.h", "PrintTo");

  *os << ::testing::PrintToString(status_or.status());
  if (status_or.ok()) {
    *os << ": " << ::testing::PrintToString(status_or.ValueOrDie());
  }
}

namespace error {
inline void PrintTo(const tensorflow::error::Code code, std::ostream* os) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_1(mht_1_v, 284, "", "./tensorflow/core/platform/status_matchers.h", "PrintTo");

  *os << Code_Name(code);
}
}  // namespace error

namespace testing {
namespace internal_status {

inline const Status& GetStatus(const Status& status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_2(mht_2_v, 295, "", "./tensorflow/core/platform/status_matchers.h", "GetStatus");
 return status; }

template <typename T>
inline const Status& GetStatus(const StatusOr<T>& status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_3(mht_3_v, 301, "", "./tensorflow/core/platform/status_matchers.h", "GetStatus");

  return status.status();
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().
//
// Monomorphic implementation of matcher IsOkAndHolds(m). StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_4(mht_4_v, 323, "", "./tensorflow/core/platform/status_matchers.h", "IsOkAndHoldsMatcherImpl");
}

  void DescribeTo(std::ostream* os) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_5(mht_5_v, 328, "", "./tensorflow/core/platform/status_matchers.h", "DescribeTo");

    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_6(mht_6_v, 336, "", "./tensorflow/core/platform/status_matchers.h", "DescribeNegationTo");

    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusOrType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_7(mht_7_v, 346, "", "./tensorflow/core/platform/status_matchers.h", "MatchAndExplain");

    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *result_listener << "which contains value "
                       << ::testing::PrintToString(*actual_value) << ", "
                       << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_8(mht_8_v, 376, "", "./tensorflow/core/platform/status_matchers.h", "IsOkAndHoldsMatcher");
}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. StatusOrType can be either StatusOr<T> or a reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_9(mht_9_v, 384, "", "./tensorflow/core/platform/status_matchers.h", "::testing::Matcher<StatusOrType>");
  // NOLINT
    return ::testing::Matcher<StatusOrType>(
        new IsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of StatusIs().
//
// StatusIs() is a polymorphic matcher. This class is the common
// implementation of it shared by all types T where StatusIs() can be used as
// a Matcher<T>.

class StatusIsMatcherCommonImpl {
 public:
  StatusIsMatcherCommonImpl(
      ::testing::Matcher<const tensorflow::error::Code> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_10(mht_10_v, 409, "", "./tensorflow/core/platform/status_matchers.h", "StatusIsMatcherCommonImpl");
}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<const tensorflow::error::Code> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher StatusIs() for a given type T. T can
// be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoStatusIsMatcherImpl(StatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_11(mht_11_v, 432, "", "./tensorflow/core/platform/status_matchers.h", "MonoStatusIsMatcherImpl");
}

  void DescribeTo(std::ostream* os) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_12(mht_12_v, 437, "", "./tensorflow/core/platform/status_matchers.h", "DescribeTo");

    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_13(mht_13_v, 444, "", "./tensorflow/core/platform/status_matchers.h", "DescribeNegationTo");

    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_14(mht_14_v, 453, "", "./tensorflow/core/platform/status_matchers.h", "MatchAndExplain");

    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  StatusIsMatcherCommonImpl common_impl_;
};

// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  StatusIsMatcher(
      ::testing::Matcher<const tensorflow::error::Code> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : common_impl_(
            ::testing::MatcherCast<const tensorflow::error::Code>(code_matcher),
            ::testing::MatcherCast<const std::string&>(message_matcher)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_15(mht_15_v, 473, "", "./tensorflow/core/platform/status_matchers.h", "StatusIsMatcher");
}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_16(mht_16_v, 481, "", "./tensorflow/core/platform/status_matchers.h", "::testing::Matcher<T>");
  // NOLINT
    return ::testing::MakeMatcher(new MonoStatusIsMatcherImpl<T>(common_impl_));
  }

 private:
  const StatusIsMatcherCommonImpl common_impl_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_17(mht_17_v, 497, "", "./tensorflow/core/platform/status_matchers.h", "DescribeTo");
 *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_18(mht_18_v, 501, "", "./tensorflow/core/platform/status_matchers.h", "DescribeNegationTo");

    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       ::testing::MatchResultListener*) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_19(mht_19_v, 508, "", "./tensorflow/core/platform/status_matchers.h", "MatchAndExplain");

    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_20(mht_20_v, 520, "", "./tensorflow/core/platform/status_matchers.h", "::testing::Matcher<T>");
  // NOLINT
    return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
  }
};
}  // namespace internal_status

// Returns a matcher that matches a StatusOr<> whose status is OK and whose
// value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename CodeMatcher, typename MessageMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher,
                                          MessageMatcher message_matcher) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_21(mht_21_v, 543, "", "./tensorflow/core/platform/status_matchers.h", "StatusIs");

  return internal_status::StatusIsMatcher(std::move(code_matcher),
                                          std::move(message_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename CodeMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_22(mht_22_v, 554, "", "./tensorflow/core/platform/status_matchers.h", "StatusIs");

  return StatusIs(std::move(code_matcher), ::testing::_);
}

// Returns a matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::IsOkMatcher IsOk() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchersDTh mht_23(mht_23_v, 562, "", "./tensorflow/core/platform/status_matchers.h", "IsOk");

  return internal_status::IsOkMatcher();
}

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
