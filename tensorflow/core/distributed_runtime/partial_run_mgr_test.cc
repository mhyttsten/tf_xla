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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/partial_run_mgr.h"

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(PartialRunMgrFindOrCreate, Create) {
  // Basic test of PartialRunMgr CancellationManager creation.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  EXPECT_TRUE(cancellation_manager != nullptr);
}

TEST(PartialRunMgrFindOrCreate, Find) {
  // Basic test of PartialRunMgr CancellationManager find.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  // Looking for the same step should return the same cancellation_manager.
  CancellationManager* found_cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &found_cancellation_manager);
  EXPECT_EQ(cancellation_manager, found_cancellation_manager);
}

TEST(PartialRunMgrFindOrCreate, NewCreate) {
  // Test that PartialRunMgr creates a new CancellationManager for new steps.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  // FindOrCreate on a new step should return a new cancellation_manager.
  int new_step_id = 2;
  CancellationManager* new_cancellation_manager;
  partial_run_mgr.FindOrCreate(new_step_id, &new_cancellation_manager);
  EXPECT_NE(cancellation_manager, new_cancellation_manager);
}

TEST(PartialRunMgr, PartialRunRemoved) {
  // Test that PartialRunMgr ensures that the PartialRun is deleted after
  // ExecutorDone and PartialRunDone are called.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);

  int called = 0;
  partial_run_mgr.PartialRunDone(
      step_id, [&called](Status status) { called++; }, Status::OK());
  partial_run_mgr.ExecutorDone(step_id, Status::OK());

  // Calling ExecutorDone and PartialRunDone on the step_id should still only
  // result in the callback being called once.
  // This proves that the original PartialRun has been removed.
  partial_run_mgr.PartialRunDone(
      step_id, [&called](Status status) { called++; }, Status::OK());
  partial_run_mgr.ExecutorDone(step_id, Status::OK());
  EXPECT_EQ(1, called);
}

struct StatusTestParam {
  Status executor_status;
  Status partial_run_status;
  Status expected_status;
};

class StatusPropagationTest : public ::testing::TestWithParam<StatusTestParam> {
 protected:
  PartialRunMgr partial_run_mgr_;

  // State to help keep track of when the callback is called.
  Notification invoked_;
  Status status_;

  void set_status(const Status& status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc mht_0(mht_0_v, 263, "", "./tensorflow/core/distributed_runtime/partial_run_mgr_test.cc", "set_status");

    status_ = status;
    invoked_.Notify();
  }

  // Blocks until status is set.
  Status status() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/distributed_runtime/partial_run_mgr_test.cc", "status");

    invoked_.WaitForNotification();
    return status_;
  }
};

TEST_P(StatusPropagationTest, ExecutorDoneFirst) {
  // Tests error propagation when ExecutorDone is called first.
  StatusTestParam param = GetParam();
  int step_id = 1;

  CancellationManager* cancellation_manager;
  partial_run_mgr_.FindOrCreate(step_id, &cancellation_manager);

  partial_run_mgr_.ExecutorDone(step_id, param.executor_status);
  partial_run_mgr_.PartialRunDone(step_id,
                                  [this](Status status) { set_status(status); },
                                  param.partial_run_status);

  EXPECT_EQ(status(), param.expected_status);
}

TEST_P(StatusPropagationTest, PartialRunDoneFirst) {
  // Tests error propagation when PartialRunDone is called first.
  StatusTestParam param = GetParam();
  int step_id = 1;

  CancellationManager* cancellation_manager;
  partial_run_mgr_.FindOrCreate(step_id, &cancellation_manager);

  partial_run_mgr_.PartialRunDone(step_id,
                                  [this](Status status) { set_status(status); },
                                  param.partial_run_status);
  partial_run_mgr_.ExecutorDone(step_id, param.executor_status);

  EXPECT_EQ(status(), param.expected_status);
}

// Instantiate tests for all error orderings, for both call orders of
// ExecutorDone and PartialRunDone.
Status ExecutorError() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc mht_2(mht_2_v, 315, "", "./tensorflow/core/distributed_runtime/partial_run_mgr_test.cc", "ExecutorError");
 return errors::Internal("executor error"); }
Status PartialRunError() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSpartial_run_mgr_testDTcc mht_3(mht_3_v, 319, "", "./tensorflow/core/distributed_runtime/partial_run_mgr_test.cc", "PartialRunError");
 return errors::Internal("partial run error"); }
INSTANTIATE_TEST_SUITE_P(
    PartialRunMgr, StatusPropagationTest,
    ::testing::Values(
        StatusTestParam{Status::OK(), Status::OK(), Status::OK()},
        StatusTestParam{ExecutorError(), Status::OK(), ExecutorError()},
        StatusTestParam{Status::OK(), PartialRunError(), PartialRunError()},
        StatusTestParam{ExecutorError(), PartialRunError(), ExecutorError()}));

}  // namespace
}  // namespace tensorflow
