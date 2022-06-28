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
class MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc() {
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
#include "tensorflow/core/platform/status_matchers.h"

#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::Ne;
using ::testing::Not;
using ::testing::PrintToString;

// Matches a value less than the given upper bound. This matcher is chatty (it
// always explains the match result with some detail), and thus is useful for
// testing that an outer matcher correctly incorporates an inner matcher's
// explanation.
MATCHER_P(LessThan, upper, "") {
  if (arg < upper) {
    *result_listener << "which is " << (upper - arg) << " less than " << upper;
    return true;
  }
  *result_listener << "which is " << (arg - upper) << " more than " << upper;
  return false;
}

// Returns the description of the given matcher.
template <typename T>
std::string Describe(const Matcher<T>& matcher) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/platform/status_matchers_test.cc", "Describe");

  std::stringstream ss;
  matcher.DescribeTo(&ss);
  return ss.str();
}

// Returns the description of the negation of the given matcher.
template <typename T>
std::string DescribeNegation(const Matcher<T>& matcher) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/platform/status_matchers_test.cc", "DescribeNegation");

  std::stringstream ss;
  matcher.DescribeNegationTo(&ss);
  return ss.str();
}

// Returns the explanation on the result of using the given matcher to
// match the given value.
template <typename T, typename V>
std::string ExplainMatch(const Matcher<T>& matcher, const V& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSstatus_matchers_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/platform/status_matchers_test.cc", "ExplainMatch");

  ::testing::StringMatchResultListener listener;
  matcher.MatchAndExplain(value, &listener);
  return listener.str();
}

TEST(IsOkAndHoldsTest, MatchesValue) {
  StatusOr<std::string> status_or_message("Hello, world");
  EXPECT_THAT(status_or_message, IsOkAndHolds("Hello, world"));
  EXPECT_THAT(status_or_message, IsOkAndHolds(HasSubstr("Hello,")));
}

TEST(IsOkAndHoldsTest, MatchesContainer) {
  StatusOr<std::vector<std::string>> status_or_messages =
      std::vector<std::string>{"Hello, world", "Hello, tf"};
  EXPECT_THAT(status_or_messages,
              IsOkAndHolds(ElementsAre("Hello, world", "Hello, tf")));
  EXPECT_THAT(status_or_messages,
              IsOkAndHolds(ElementsAre(HasSubstr("world"), HasSubstr("tf"))));
}

TEST(IsOkAndHoldsTest, DoesNotMatchStatus) {
  StatusOr<std::string> status_or_message =
      errors::InvalidArgument("Invalid argument");
  EXPECT_THAT(status_or_message, Not(IsOkAndHolds("Hello, world")));
}

TEST(IsOkAndHoldsTest, DoesNotMatchValue) {
  StatusOr<std::string> status_or_message("Hello, tf");
  EXPECT_THAT(status_or_message, Not(IsOkAndHolds("Hello, world")));
}

TEST(IsOkAndHoldsTest, DoesNotMatchContainer) {
  StatusOr<std::vector<int>> status_or_container({1, 2, 3});
  EXPECT_THAT(status_or_container, Not(IsOkAndHolds(ElementsAre(4, 5, 6))));
}

TEST(IsOkAndHoldsTest, DescribeExpectedValue) {
  Matcher<StatusOr<std::string>> is_ok_and_has_substr =
      IsOkAndHolds(HasSubstr("Hello"));
  EXPECT_EQ(Describe(is_ok_and_has_substr),
            "is OK and has a value that has substring \"Hello\"");
  EXPECT_EQ(DescribeNegation(is_ok_and_has_substr),
            "isn't OK or has a value that has no substring \"Hello\"");
}

TEST(IsOkAndHoldsTest, ExplainNotMatchingStatus) {
  Matcher<StatusOr<int>> is_ok_and_less_than = IsOkAndHolds(LessThan(100));
  StatusOr<int> status = errors::Unknown("Unknown");
  EXPECT_EQ(ExplainMatch(is_ok_and_less_than, status),
            "which has status " + PrintToString(status));
}

TEST(IsOkAndHoldsTest, ExplainNotMatchingValue) {
  Matcher<StatusOr<int>> is_ok_and_less_than = IsOkAndHolds(LessThan(100));
  EXPECT_EQ(ExplainMatch(is_ok_and_less_than, 120),
            "which contains value 120, which is 20 more than 100");
}

TEST(IsOkAndHoldsTest, ExplainNotMatchingContainer) {
  Matcher<StatusOr<std::vector<int>>> is_ok_and_less_than =
      IsOkAndHolds(ElementsAre(1, 2, 3));
  std::vector<int> actual{4, 5, 6};
  EXPECT_THAT(ExplainMatch(is_ok_and_less_than, actual),
              HasSubstr("which contains value " + PrintToString(actual)));
}

TEST(StatusIsTest, MatchesOK) {
  EXPECT_THAT(Status::OK(), StatusIs(error::OK));
  StatusOr<std::string> message("Hello, world");
  EXPECT_THAT(message, StatusIs(error::OK));
}

TEST(StatusIsTest, DoesNotMatchOk) {
  EXPECT_THAT(errors::DeadlineExceeded("Deadline exceeded"),
              Not(StatusIs(error::OK)));
  StatusOr<std::string> status = errors::NotFound("Not found");
  EXPECT_THAT(status, Not(StatusIs(error::OK)));
}

TEST(StatusIsTest, MatchesStatus) {
  Status s = errors::Cancelled("Cancelled");
  EXPECT_THAT(s, StatusIs(error::CANCELLED));
  EXPECT_THAT(s, StatusIs(error::CANCELLED, "Cancelled"));
  EXPECT_THAT(s, StatusIs(_, "Cancelled"));
  EXPECT_THAT(s, StatusIs(error::CANCELLED, _));
  EXPECT_THAT(s, StatusIs(Ne(error::INVALID_ARGUMENT), _));
  EXPECT_THAT(s, StatusIs(error::CANCELLED, HasSubstr("Can")));
  EXPECT_THAT(s, StatusIs(error::CANCELLED, MatchesRegex("Can.*")));
}

TEST(StatusIsTest, StatusOrMatchesStatus) {
  StatusOr<int> s = errors::InvalidArgument("Invalid Argument");
  EXPECT_THAT(s, StatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(s, StatusIs(error::INVALID_ARGUMENT, "Invalid Argument"));
  EXPECT_THAT(s, StatusIs(_, "Invalid Argument"));
  EXPECT_THAT(s, StatusIs(error::INVALID_ARGUMENT, _));
  EXPECT_THAT(s, StatusIs(Ne(error::CANCELLED), _));
  EXPECT_THAT(s, StatusIs(error::INVALID_ARGUMENT, HasSubstr("Argument")));
  EXPECT_THAT(s, StatusIs(error::INVALID_ARGUMENT, MatchesRegex(".*Argument")));
}

TEST(StatusIsTest, DoesNotMatchStatus) {
  Status s = errors::Internal("Internal");
  EXPECT_THAT(s, Not(StatusIs(error::FAILED_PRECONDITION)));
  EXPECT_THAT(s, Not(StatusIs(error::INTERNAL, "Failed Precondition")));
  EXPECT_THAT(s, Not(StatusIs(_, "Failed Precondition")));
  EXPECT_THAT(s, Not(StatusIs(error::FAILED_PRECONDITION, _)));
}

TEST(StatusIsTest, StatusOrDoesNotMatchStatus) {
  StatusOr<int> s = errors::FailedPrecondition("Failed Precondition");
  EXPECT_THAT(s, Not(StatusIs(error::INTERNAL)));
  EXPECT_THAT(s, Not(StatusIs(error::FAILED_PRECONDITION, "Internal")));
  EXPECT_THAT(s, Not(StatusIs(_, "Internal")));
  EXPECT_THAT(s, Not(StatusIs(error::INTERNAL, _)));
}

TEST(StatusIsTest, DescribeExpectedValue) {
  Matcher<Status> status_is =
      StatusIs(error::UNAVAILABLE, std::string("Unavailable"));
  EXPECT_EQ(Describe(status_is),
            "has a status code that is equal to UNAVAILABLE, "
            "and has an error message that is equal to \"Unavailable\"");
}

TEST(StatusIsTest, DescribeNegatedExpectedValue) {
  Matcher<StatusOr<std::string>> status_is =
      StatusIs(error::ABORTED, std::string("Aborted"));
  EXPECT_EQ(DescribeNegation(status_is),
            "has a status code that isn't equal to ABORTED, "
            "or has an error message that isn't equal to \"Aborted\"");
}

TEST(StatusIsTest, ExplainNotMatchingErrorCode) {
  Matcher<Status> status_is = StatusIs(error::NOT_FOUND, _);
  const Status status = errors::AlreadyExists("Already exists");
  EXPECT_EQ(ExplainMatch(status_is, status), "whose status code is wrong");
}

TEST(StatusIsTest, ExplainNotMatchingErrorMessage) {
  Matcher<Status> status_is = StatusIs(error::NOT_FOUND, "Not found");
  const Status status = errors::NotFound("Already exists");
  EXPECT_EQ(ExplainMatch(status_is, status), "whose error message is wrong");
}

TEST(StatusIsTest, ExplainStatusOrNotMatchingErrorCode) {
  Matcher<StatusOr<int>> status_is = StatusIs(error::ALREADY_EXISTS, _);
  const StatusOr<int> status_or = errors::NotFound("Not found");
  EXPECT_EQ(ExplainMatch(status_is, status_or), "whose status code is wrong");
}

TEST(StatusIsTest, ExplainStatusOrNotMatchingErrorMessage) {
  Matcher<StatusOr<int>> status_is =
      StatusIs(error::ALREADY_EXISTS, "Already exists");
  const StatusOr<int> status_or = errors::AlreadyExists("Not found");
  EXPECT_EQ(ExplainMatch(status_is, status_or), "whose error message is wrong");
}

TEST(StatusIsTest, ExplainStatusOrHasValue) {
  Matcher<StatusOr<int>> status_is =
      StatusIs(error::RESOURCE_EXHAUSTED, "Resource exhausted");
  const StatusOr<int> value = -1;
  EXPECT_EQ(ExplainMatch(status_is, value), "whose status code is wrong");
}

TEST(IsOkTest, MatchesOK) {
  EXPECT_THAT(Status::OK(), IsOk());
  StatusOr<std::string> message = std::string("Hello, world");
  EXPECT_THAT(message, IsOk());
}

TEST(IsOkTest, DoesNotMatchOK) {
  EXPECT_THAT(errors::PermissionDenied("Permission denied"), Not(IsOk()));
  StatusOr<std::string> status = errors::Unauthenticated("Unauthenticated");
  EXPECT_THAT(status, Not(IsOk()));
}

TEST(IsOkTest, DescribeExpectedValue) {
  Matcher<Status> status_is_ok = IsOk();
  EXPECT_EQ(Describe(status_is_ok), "is OK");
  Matcher<StatusOr<std::string>> status_or_is_ok = IsOk();
  EXPECT_EQ(Describe(status_or_is_ok), "is OK");
}

TEST(IsOkTest, DescribeNegatedExpectedValue) {
  Matcher<Status> status_is_ok = IsOk();
  EXPECT_EQ(DescribeNegation(status_is_ok), "is not OK");
  Matcher<StatusOr<std::string>> status_or_is_ok = IsOk();
  EXPECT_EQ(DescribeNegation(status_or_is_ok), "is not OK");
}

}  // namespace
}  // namespace testing
}  // namespace tensorflow
