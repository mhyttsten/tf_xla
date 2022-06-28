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
class MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc() {
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

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Status RetCheckFail() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/status_macros_test.cc", "RetCheckFail");

  TF_RET_CHECK(2 > 3);
  return Status::OK();
}

Status RetCheckFailWithExtraMessage() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_1(mht_1_v, 202, "", "./tensorflow/compiler/xla/status_macros_test.cc", "RetCheckFailWithExtraMessage");

  TF_RET_CHECK(2 > 3) << "extra message";
  return Status::OK();
}

Status RetCheckSuccess() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_2(mht_2_v, 210, "", "./tensorflow/compiler/xla/status_macros_test.cc", "RetCheckSuccess");

  TF_RET_CHECK(3 > 2);
  return Status::OK();
}

TEST(StatusMacros, RetCheckFailing) {
  Status status = RetCheckFail();
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(status.error_message(),
              ::testing::ContainsRegex("RET_CHECK failure.*2 > 3"));
}

TEST(StatusMacros, RetCheckFailingWithExtraMessage) {
  Status status = RetCheckFailWithExtraMessage();
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(status.error_message(),
              ::testing::ContainsRegex("RET_CHECK.*2 > 3 extra message"));
}

TEST(StatusMacros, RetCheckSucceeding) {
  Status status = RetCheckSuccess();
  EXPECT_IS_OK(status);
}

StatusOr<int> CreateIntSuccessfully() { return 42; }

StatusOr<int> CreateIntUnsuccessfully() {
  return tensorflow::errors::Internal("foobar");
}

TEST(StatusMacros, AssignOrAssertOnOK) {
  TF_ASSERT_OK_AND_ASSIGN(int result, CreateIntSuccessfully());
  EXPECT_EQ(42, result);
}

Status ReturnStatusOK() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_3(mht_3_v, 248, "", "./tensorflow/compiler/xla/status_macros_test.cc", "ReturnStatusOK");
 return Status::OK(); }

Status ReturnStatusError() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/xla/status_macros_test.cc", "ReturnStatusError");
 return (tensorflow::errors::Internal("foobar")); }

using StatusReturningFunction = std::function<Status()>;

StatusOr<int> CallStatusReturningFunction(const StatusReturningFunction& func) {
  TF_RETURN_IF_ERROR(func());
  return 42;
}

TEST(StatusMacros, ReturnIfErrorOnOK) {
  StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusOK);
  EXPECT_IS_OK(rc);
  EXPECT_EQ(42, rc.ConsumeValueOrDie());
}

TEST(StatusMacros, ReturnIfErrorOnError) {
  StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusError);
  EXPECT_FALSE(rc.ok());
  EXPECT_EQ(rc.status().code(), tensorflow::error::INTERNAL);
}

TEST(StatusMacros, AssignOrReturnSuccessfully) {
  Status status = []() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/xla/status_macros_test.cc", "lambda");

    TF_ASSIGN_OR_RETURN(int value, CreateIntSuccessfully());
    EXPECT_EQ(value, 42);
    return Status::OK();
  }();
  EXPECT_IS_OK(status);
}

TEST(StatusMacros, AssignOrReturnUnsuccessfully) {
  Status status = []() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSstatus_macros_testDTcc mht_6(mht_6_v, 290, "", "./tensorflow/compiler/xla/status_macros_test.cc", "lambda");

    TF_ASSIGN_OR_RETURN(int value, CreateIntUnsuccessfully());
    (void)value;
    return Status::OK();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
}

}  // namespace xla
