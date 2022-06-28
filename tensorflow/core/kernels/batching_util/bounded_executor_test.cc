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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc() {
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

#include "tensorflow/core/kernels/batching_util/bounded_executor.h"

#include "absl/functional/bind_front.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace serving {

namespace {
// Tracks the number of concurrently running tasks.
class TaskTracker {
 public:
  // Creates a functor that invokes Run() with the given arguments.
  std::function<void()> MakeTask(int task_id, absl::Duration sleep_duration) {
    return absl::bind_front(&TaskTracker::Run, this, task_id, sleep_duration);
  }

  // Updates run counts, sleeps for a short time and then returns.
  // Exits early if fiber is cancelled.
  void Run(int task_id, absl::Duration sleep_duration) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/batching_util/bounded_executor_test.cc", "Run");

    LOG(INFO) << "Entering task " << task_id;
    // Update run counters.
    {
      mutex_lock l(mutex_);
      ++task_count_;
      ++running_count_;
      if (running_count_ > max_running_count_) {
        max_running_count_ = running_count_;
      }
    }

    // Use a sleep loop so we can quickly detect cancellation even when the
    // total sleep time is very large.

    Env::Default()->SleepForMicroseconds(
        absl::ToInt64Microseconds(sleep_duration));
    // Update run counters.
    {
      mutex_lock l(mutex_);
      --running_count_;
    }
    LOG(INFO) << "Task " << task_id << " exiting.";
  }

  // Returns number of tasks that have been run.
  int task_count() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/batching_util/bounded_executor_test.cc", "task_count");

    mutex_lock l(mutex_);
    return task_count_;
  }

  // Returns number of tasks that are currently running.
  int running_count() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/batching_util/bounded_executor_test.cc", "running_count");

    mutex_lock l(mutex_);
    return running_count_;
  }

  // Returns the max number of tasks that have run concurrently.
  int max_running_count() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbounded_executor_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/kernels/batching_util/bounded_executor_test.cc", "max_running_count");

    mutex_lock l(mutex_);
    return max_running_count_;
  }

 private:
  mutex mutex_;
  int task_count_ = 0;
  int running_count_ = 0;
  int max_running_count_ = 0;
};

TEST(BoundedExecutorTest, InvalidEmptyEnv) {
  BoundedExecutor::Options options;
  options.num_threads = 2;
  options.env = nullptr;
  EXPECT_THAT(BoundedExecutor::Create(options),
              ::tensorflow::testing::StatusIs(
                  error::INVALID_ARGUMENT, "options.env must not be nullptr"));
}

TEST(BoundedExecutorTest, InvalidNumThreads) {
  {
    BoundedExecutor::Options options;
    options.num_threads = 0;
    EXPECT_THAT(
        BoundedExecutor::Create(options),
        ::tensorflow::testing::StatusIs(
            error::INVALID_ARGUMENT, "options.num_threads must be positive"));
  }

  {
    BoundedExecutor::Options options;
    options.num_threads = -1;
    EXPECT_THAT(
        BoundedExecutor::Create(options),
        ::tensorflow::testing::StatusIs(
            error::INVALID_ARGUMENT, "options.num_threads must be positive"));
  }
}

TEST(BoundedExecutorTest, AddRunsFunctionsEventually) {
  BoundedExecutor::Options options;
  options.num_threads = 2;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, BoundedExecutor::Create(options));

  Notification done0;
  executor->Schedule([&done0] { done0.Notify(); });
  Notification done1;
  executor->Schedule([&done1] { done1.Notify(); });
  done0.WaitForNotification();
  done1.WaitForNotification();

  executor.reset();
}

TEST(BoundedExecutorTest, MaxInflightLimit) {
  BoundedExecutor::Options options;
  options.num_threads = 5;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, BoundedExecutor::Create(options));

  const int num_tasks = 100;
  TaskTracker task_tracker;
  for (int i = 0; i < num_tasks; i++) {
    executor->Schedule(task_tracker.MakeTask(i, absl::Seconds(1)));
  }
  executor.reset();

  EXPECT_EQ(task_tracker.task_count(), num_tasks);
  EXPECT_EQ(task_tracker.max_running_count(), options.num_threads);
  EXPECT_EQ(task_tracker.running_count(), 0);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
