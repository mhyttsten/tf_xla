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
class MHTracer_DTPStensorflowPScorePSlibPScorePSstatus_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSstatus_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSstatus_testDTcc() {
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

#include "tensorflow/core/lib/core/status.h"

#include "absl/strings/match.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Status, OK) {
  EXPECT_EQ(Status::OK().code(), error::OK);
  EXPECT_EQ(Status::OK().error_message(), "");
  TF_EXPECT_OK(Status::OK());
  TF_ASSERT_OK(Status::OK());
  EXPECT_EQ(Status::OK(), Status());
  Status s;
  EXPECT_TRUE(s.ok());
}

TEST(DeathStatus, CheckOK) {
  Status status(errors::InvalidArgument("Invalid"));
  ASSERT_DEATH(TF_CHECK_OK(status), "Invalid");
}

TEST(Status, Set) {
  Status status;
  status = Status(error::CANCELLED, "Error message");
  EXPECT_EQ(status.code(), error::CANCELLED);
  EXPECT_EQ(status.error_message(), "Error message");
}

TEST(Status, Copy) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Move) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(std::move(a));
  ASSERT_EQ("INVALID_ARGUMENT: Invalid", b.ToString());
}

TEST(Status, MoveAssign) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b;
  b = std::move(a);
  ASSERT_EQ("INVALID_ARGUMENT: Invalid", b.ToString());
}

TEST(Status, Update) {
  Status s;
  s.Update(Status::OK());
  ASSERT_TRUE(s.ok());
  Status a(errors::InvalidArgument("Invalid"));
  s.Update(a);
  ASSERT_EQ(s.ToString(), a.ToString());
  Status b(errors::Internal("Internal"));
  s.Update(b);
  ASSERT_EQ(s.ToString(), a.ToString());
  s.Update(Status::OK());
  ASSERT_EQ(s.ToString(), a.ToString());
  ASSERT_FALSE(s.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(Status::OK(), Status()); }

TEST(Status, EqualsSame) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(errors::InvalidArgument("Invalid"));
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const Status a(errors::InvalidArgument("Invalid"));
  const Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::Internal("message"));
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::InvalidArgument("another"));
  ASSERT_NE(a, b);
}

TEST(StatusGroup, OKStatusGroup) {
  StatusGroup c;
  c.Update(Status::OK());
  c.Update(Status::OK());
  ASSERT_EQ(c.as_summary_status(), Status::OK());
  ASSERT_EQ(c.as_concatenated_status(), Status::OK());
}

TEST(StatusGroup, AggregateWithSingleErrorStatus) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));

  c.Update(internal);
  ASSERT_EQ(c.as_summary_status(), internal);

  Status concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));

  // Add derived error status
  const Status derived =
      StatusGroup::MakeDerived(errors::Internal("Derived error."));
  c.Update(derived);

  ASSERT_EQ(c.as_summary_status(), internal);

  concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));
}

TEST(StatusGroup, AggregateWithMultipleErrorStatus) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));
  const Status cancelled(errors::Cancelled("Cancelled after 10 steps."));
  const Status aborted(errors::Aborted("Aborted after 10 steps."));

  c.Update(internal);
  c.Update(cancelled);
  c.Update(aborted);

  Status summary = c.as_summary_status();

  ASSERT_EQ(summary.code(), internal.code());
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), internal.error_message()));
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), cancelled.error_message()));
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), aborted.error_message()));

  Status concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                cancelled.error_message()));
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                aborted.error_message()));
}

TEST(Status, InvalidPayloadGetsIgnored) {
  Status s = Status();
  s.SetPayload("Invalid", "Invalid Val");
  ASSERT_FALSE(s.GetPayload("Invalid").has_value());
  bool is_err_erased = s.ErasePayload("Invalid");
  ASSERT_EQ(is_err_erased, false);
}

TEST(Status, SetPayloadSetsOrUpdatesIt) {
  Status s(error::INTERNAL, "Error message");
  s.SetPayload("Error key", "Original");
  ASSERT_EQ(s.GetPayload("Error key"), tensorflow::StringPiece("Original"));
  s.SetPayload("Error key", "Updated");
  ASSERT_EQ(s.GetPayload("Error key"), tensorflow::StringPiece("Updated"));
}

TEST(Status, ErasePayloadRemovesIt) {
  Status s(error::INTERNAL, "Error message");
  s.SetPayload("Error key", "Original");

  bool is_err_erased = s.ErasePayload("Error key");
  ASSERT_EQ(is_err_erased, true);
  is_err_erased = s.ErasePayload("Error key");
  ASSERT_EQ(is_err_erased, false);
  ASSERT_FALSE(s.GetPayload("Error key").has_value());
}

static void BM_TF_CHECK_OK(::testing::benchmark::State& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSstatus_testDTcc mht_0(mht_0_v, 374, "", "./tensorflow/core/lib/core/status_test.cc", "BM_TF_CHECK_OK");

  tensorflow::Status s = (state.max_iterations < 0)
                             ? errors::InvalidArgument("Invalid")
                             : Status::OK();
  for (auto i : state) {
    TF_CHECK_OK(s);
  }
}
BENCHMARK(BM_TF_CHECK_OK);

}  // namespace tensorflow
