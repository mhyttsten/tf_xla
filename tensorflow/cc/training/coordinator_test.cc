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
class MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc {
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
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc() {
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

#include "tensorflow/cc/training/coordinator.h"

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

using error::Code;

void WaitForStopThread(Coordinator* coord, Notification* about_to_wait,
                       Notification* done) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/cc/training/coordinator_test.cc", "WaitForStopThread");

  about_to_wait->Notify();
  coord->WaitForStop();
  done->Notify();
}

TEST(CoordinatorTest, TestStopAndWaitOnStop) {
  Coordinator coord;
  EXPECT_EQ(coord.ShouldStop(), false);

  Notification about_to_wait;
  Notification done;
  Env::Default()->SchedClosure(
      std::bind(&WaitForStopThread, &coord, &about_to_wait, &done));
  about_to_wait.WaitForNotification();
  Env::Default()->SleepForMicroseconds(1000 * 1000);
  EXPECT_FALSE(done.HasBeenNotified());

  TF_EXPECT_OK(coord.RequestStop());
  done.WaitForNotification();
  EXPECT_TRUE(coord.ShouldStop());
}

class MockQueueRunner : public RunnerInterface {
 public:
  explicit MockQueueRunner(Coordinator* coord) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/cc/training/coordinator_test.cc", "MockQueueRunner");

    coord_ = coord;
    join_counter_ = nullptr;
    thread_pool_.reset(new thread::ThreadPool(Env::Default(), "test-pool", 10));
    stopped_ = false;
  }

  MockQueueRunner(Coordinator* coord, int* join_counter)
      : MockQueueRunner(coord) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/cc/training/coordinator_test.cc", "MockQueueRunner");

    join_counter_ = join_counter;
  }

  void StartCounting(std::atomic<int>* counter, int until,
                     Notification* start = nullptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_3(mht_3_v, 249, "", "./tensorflow/cc/training/coordinator_test.cc", "StartCounting");

    thread_pool_->Schedule(
        std::bind(&MockQueueRunner::CountThread, this, counter, until, start));
  }

  void StartSettingStatus(const Status& status, BlockingCounter* counter,
                          Notification* start) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_4(mht_4_v, 258, "", "./tensorflow/cc/training/coordinator_test.cc", "StartSettingStatus");

    thread_pool_->Schedule(std::bind(&MockQueueRunner::SetStatusThread, this,
                                     status, counter, start));
  }

  Status Join() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_5(mht_5_v, 266, "", "./tensorflow/cc/training/coordinator_test.cc", "Join");

    if (join_counter_ != nullptr) {
      (*join_counter_)++;
    }
    thread_pool_.reset();
    return status_;
  }

  Status GetStatus() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_6(mht_6_v, 277, "", "./tensorflow/cc/training/coordinator_test.cc", "GetStatus");
 return status_; }

  void SetStatus(const Status& status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_7(mht_7_v, 282, "", "./tensorflow/cc/training/coordinator_test.cc", "SetStatus");
 status_ = status; }

  bool IsRunning() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_8(mht_8_v, 287, "", "./tensorflow/cc/training/coordinator_test.cc", "IsRunning");
 return !stopped_; };

  void Stop() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_9(mht_9_v, 292, "", "./tensorflow/cc/training/coordinator_test.cc", "Stop");
 stopped_ = true; }

 private:
  void CountThread(std::atomic<int>* counter, int until, Notification* start) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_10(mht_10_v, 298, "", "./tensorflow/cc/training/coordinator_test.cc", "CountThread");

    if (start != nullptr) start->WaitForNotification();
    while (!coord_->ShouldStop() && counter->load() < until) {
      (*counter)++;
      Env::Default()->SleepForMicroseconds(10 * 1000);
    }
    coord_->RequestStop().IgnoreError();
  }
  void SetStatusThread(const Status& status, BlockingCounter* counter,
                       Notification* start) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinator_testDTcc mht_11(mht_11_v, 310, "", "./tensorflow/cc/training/coordinator_test.cc", "SetStatusThread");

    start->WaitForNotification();
    SetStatus(status);
    counter->DecrementCount();
  }
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  Status status_;
  Coordinator* coord_;
  int* join_counter_;
  bool stopped_;
};

TEST(CoordinatorTest, TestRealStop) {
  std::atomic<int> counter(0);
  Coordinator coord;

  std::unique_ptr<MockQueueRunner> qr1(new MockQueueRunner(&coord));
  qr1->StartCounting(&counter, 100);
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr1)));

  std::unique_ptr<MockQueueRunner> qr2(new MockQueueRunner(&coord));
  qr2->StartCounting(&counter, 100);
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr2)));

  // Wait until the counting has started
  while (counter.load() == 0)
    ;
  TF_EXPECT_OK(coord.RequestStop());

  int temp_counter = counter.load();
  Env::Default()->SleepForMicroseconds(1000 * 1000);
  EXPECT_EQ(temp_counter, counter.load());
  TF_EXPECT_OK(coord.Join());
}

TEST(CoordinatorTest, TestRequestStop) {
  Coordinator coord;
  std::atomic<int> counter(0);
  Notification start;
  std::unique_ptr<MockQueueRunner> qr;
  for (int i = 0; i < 10; i++) {
    qr.reset(new MockQueueRunner(&coord));
    qr->StartCounting(&counter, 10, &start);
    TF_ASSERT_OK(coord.RegisterRunner(std::move(qr)));
  }
  start.Notify();

  coord.WaitForStop();
  EXPECT_EQ(coord.ShouldStop(), true);
  EXPECT_EQ(counter.load(), 10);
  TF_EXPECT_OK(coord.Join());
}

TEST(CoordinatorTest, TestJoin) {
  Coordinator coord;
  int join_counter = 0;
  std::unique_ptr<MockQueueRunner> qr1(
      new MockQueueRunner(&coord, &join_counter));
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr1)));
  std::unique_ptr<MockQueueRunner> qr2(
      new MockQueueRunner(&coord, &join_counter));
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr2)));

  TF_EXPECT_OK(coord.RequestStop());
  TF_EXPECT_OK(coord.Join());
  EXPECT_EQ(join_counter, 2);
}

TEST(CoordinatorTest, StatusReporting) {
  Coordinator coord({Code::CANCELLED, Code::OUT_OF_RANGE});
  Notification start;
  BlockingCounter counter(3);

  std::unique_ptr<MockQueueRunner> qr1(new MockQueueRunner(&coord));
  qr1->StartSettingStatus(Status(Code::CANCELLED, ""), &counter, &start);
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr1)));

  std::unique_ptr<MockQueueRunner> qr2(new MockQueueRunner(&coord));
  qr2->StartSettingStatus(Status(Code::INVALID_ARGUMENT, ""), &counter, &start);
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr2)));

  std::unique_ptr<MockQueueRunner> qr3(new MockQueueRunner(&coord));
  qr3->StartSettingStatus(Status(Code::OUT_OF_RANGE, ""), &counter, &start);
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr3)));

  start.Notify();
  counter.Wait();
  TF_EXPECT_OK(coord.RequestStop());
  EXPECT_EQ(coord.Join().code(), Code::INVALID_ARGUMENT);
}

TEST(CoordinatorTest, JoinWithoutStop) {
  Coordinator coord;
  std::unique_ptr<MockQueueRunner> qr(new MockQueueRunner(&coord));
  TF_ASSERT_OK(coord.RegisterRunner(std::move(qr)));

  EXPECT_EQ(coord.Join().code(), Code::FAILED_PRECONDITION);
}

TEST(CoordinatorTest, AllRunnersStopped) {
  Coordinator coord;
  MockQueueRunner* qr = new MockQueueRunner(&coord);
  TF_ASSERT_OK(coord.RegisterRunner(std::unique_ptr<RunnerInterface>(qr)));

  EXPECT_FALSE(coord.AllRunnersStopped());
  qr->Stop();
  EXPECT_TRUE(coord.AllRunnersStopped());
}

}  // namespace
}  // namespace tensorflow
