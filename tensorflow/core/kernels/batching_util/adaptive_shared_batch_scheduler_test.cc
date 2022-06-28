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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc() {
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

#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"

#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace anonymous {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "FakeTask");
}

  ~FakeTask() override = default;

  size_t size() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "size");
 return size_; }

  void set_size(size_t size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "set_size");
 size_ = size; }

 private:
  size_t size_;

  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

// Creates a FakeTask of size 'task_size', and calls 'scheduler->Schedule()' on
// that task. Returns the resulting status.
Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "ScheduleTask");

  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

// Creates a thread that waits on 'start' and then advances the fake clock in
// 'env' in a loop until 'stop' is notified. Useful for allowing objects that
// use the clock to be destroyed.
std::unique_ptr<Thread> CreateFakeClockAdvancerThread(
    test_util::FakeClockEnv* env, Notification* start, Notification* stop) {
  return std::unique_ptr<Thread>(Env::Default()->StartThread(
      {}, "FakeClockAdvancerThread", [env, start, stop] {
        start->WaitForNotification();
        while (!stop->HasBeenNotified()) {
          env->AdvanceByMicroseconds(10);
          Env::Default()->SleepForMicroseconds(10);
        }
      }));
}

TEST(AdaptiveSharedBatchSchedulerTest, BadOptions) {
  using Scheduler = AdaptiveSharedBatchScheduler<FakeTask>;
  std::shared_ptr<Scheduler> scheduler;
  Scheduler::Options options;
  options.num_batch_threads = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.initial_in_flight_batches_limit = 0.5;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.num_batch_threads = 5;
  options.initial_in_flight_batches_limit = 8;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.batches_to_average_over = -5;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_in_flight_batches_limit = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_in_flight_batches_limit = 5;
  options.num_batch_threads = 3;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.initial_in_flight_batches_limit = 1;
  options.min_in_flight_batches_limit = 2;
  options.num_batch_threads = 3;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesLimit) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 2;
  options.batches_to_average_over = 1000;
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_4(mht_4_v, 289, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    mu.lock();
    int batch_num = ++processed_batches;
    mu.unlock();
    if (batch_num == 2) {
      // Give third batch a chance to process if it's going to.
      Env::Default()->SleepForMicroseconds(1000);
      finish_processing.Notify();
    }
    if (batch_num == 3) {
      ASSERT_TRUE(finish_processing.HasBeenNotified());
    }
    finish_processing.WaitForNotification();
  };
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 3 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesLimitTuning) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 2;
    options.batches_to_average_over = 1;
    auto queue_callback = [&env](std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_5(mht_5_v, 334, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

      ASSERT_TRUE(batch->IsClosed());
      switch (batch->size()) {
        case 0:
          env.AdvanceByMicroseconds(10);
          break;
        case 1:
          env.AdvanceByMicroseconds(15);
          break;
        case 2:
          env.AdvanceByMicroseconds(10);
          break;
        case 3:
          env.AdvanceByMicroseconds(11);
          break;
      }
    };
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

    TF_ASSERT_OK(ScheduleTask(0, queue.get()));
    double in_flight_batches_limit = 2;
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Initial direction will be negative.
    EXPECT_LT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency increased -> change direction.
    EXPECT_GT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency decreased -> keep going in same direction.
    EXPECT_GT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(3, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency increased -> change direction.
    EXPECT_LT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FullBatchSchedulingBoostMicros) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 1;
    options.num_batch_threads = 1;
    options.batches_to_average_over = 1000;
    options.full_batch_scheduling_boost_micros = 100;
    mutex mu;
    int processed_batches = 0;
    Notification finish_processing;
    auto queue_callback = [&mu, &processed_batches, &finish_processing](
                              std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_6(mht_6_v, 405, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

      ASSERT_TRUE(batch->IsClosed());
      finish_processing.WaitForNotification();
      mutex_lock l(mu);
      processed_batches++;
      switch (processed_batches) {
        case 1:
          EXPECT_EQ(100, batch->size());
          break;
        case 2:
          EXPECT_EQ(50, batch->size());
          break;
        case 3:
          EXPECT_EQ(900, batch->size());
          break;
        case 4:
          EXPECT_EQ(200, batch->size());
          break;
        default:
          EXPECT_TRUE(false) << "Should only have 4 batches";
      }
    };
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    std::unique_ptr<BatchScheduler<FakeTask>> queue1;
    std::unique_ptr<BatchScheduler<FakeTask>> queue2;
    queue_options.max_batch_size = 1000;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue1));
    queue_options.max_batch_size = 100;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue2));

    // First batch immediately processed.
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    while (queue1->NumEnqueuedTasks() > 0) {
    }

    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);

    TF_ASSERT_OK(ScheduleTask(50, queue2.get()));
    env.AdvanceByMicroseconds(45);

    TF_ASSERT_OK(ScheduleTask(900, queue1.get()));

    // Second batch - creation time: 0, fullness: 0.2, sched score: -20
    // Third batch - creation time: 20, fullness: 0.5, sched score: -30
    // Fourth batch - creation time: 65, fullness: 0.9, sched score: -25

    finish_processing.Notify();
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FIFO) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 1;
    options.num_batch_threads = 1;
    options.batches_to_average_over = 1000;
    options.full_batch_scheduling_boost_micros = 0;
    options.fifo_scheduling = true;
    mutex mu;
    int processed_batches = 0;
    Notification finish_processing;
    auto queue_callback = [&mu, &processed_batches, &finish_processing](
                              std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_7(mht_7_v, 483, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

      ASSERT_TRUE(batch->IsClosed());
      finish_processing.WaitForNotification();
      mutex_lock l(mu);
      processed_batches++;
      switch (processed_batches) {
        case 1:
          EXPECT_EQ(100, batch->size());
          break;
        case 2:
          EXPECT_EQ(200, batch->size());
          break;
        case 3:
          EXPECT_EQ(50, batch->size());
          break;
        case 4:
          EXPECT_EQ(900, batch->size());
          break;
        default:
          EXPECT_TRUE(false) << "Should only have 4 batches";
      }
    };
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    std::unique_ptr<BatchScheduler<FakeTask>> queue1;
    std::unique_ptr<BatchScheduler<FakeTask>> queue2;
    queue_options.max_batch_size = 1000;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue1));
    queue_options.max_batch_size = 100;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue2));

    // First batch immediately processed.
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(30);

    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);

    TF_ASSERT_OK(ScheduleTask(50, queue2.get()));
    env.AdvanceByMicroseconds(45);

    TF_ASSERT_OK(ScheduleTask(900, queue1.get()));

    finish_processing.Notify();
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, DeleteQueue) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.num_batch_threads = 1;
  options.batches_to_average_over = 1000;
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_8(mht_8_v, 548, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    finish_processing.WaitForNotification();
    mu.lock();
    processed_batches++;
    mu.unlock();
  };

  auto processed_checker = gtl::MakeCleanup([&mu, &processed_batches] {
    mutex_lock l(mu);
    EXPECT_EQ(processed_batches, 2);
  });
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // Queue destructor should block until second batch has been scheduled.
  Env::Default()->SchedClosureAfter(
      1000, [&finish_processing] { finish_processing.Notify(); });
}

TEST(AdaptiveSharedBatchSchedulerTest, QueueCapacityInfo) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1000;
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_9(mht_9_v, 588, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    mu.lock();
    int batch_num = ++processed_batches;
    mu.unlock();
    if (batch_num == 1) {
      finish_processing.WaitForNotification();
    }
  };
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // First batch was immediately processed, no longer counts as enqueued.
  EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 900);
  // Enqueue 2 more tasks, should fall in same batch.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  TF_ASSERT_OK(ScheduleTask(200, queue.get()));
  EXPECT_EQ(queue->NumEnqueuedTasks(), 3);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 600);
  // Enqueue 1 more task, should create new batch and start processing the
  // previous batch.
  TF_ASSERT_OK(ScheduleTask(700, queue.get()));
  EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 300);
  finish_processing.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FullBatches) {
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));
  auto queue_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_10(mht_10_v, 631, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

    ASSERT_TRUE(batch->IsClosed());
  };
  AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.max_batch_size = 100;
  queue_options.batch_timeout_micros = 1000000000000;
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // Full batches should not have to wait batch_timeout_micros.
}

TEST(AdaptiveSharedBatchSchedulerTest, TruncateBatches) {
  mutex mu;
  int processed_batches = 0;
  auto queue_callback =
      [&mu, &processed_batches](std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_11(mht_11_v, 650, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

        ASSERT_TRUE(batch->IsClosed());
        mutex_lock l(mu);
        ++processed_batches;
      };
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;

  AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.max_batch_size = 100;
  queue_options.batch_timeout_micros = 1000000;
  queue_options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int first_size, int max_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_12(mht_12_v, 667, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

        EXPECT_EQ(first_size, 70);
        output_tasks->push_back(std::move(*input_task));
        int remaining_size = output_tasks->back()->size() - first_size;
        output_tasks->back()->set_size(first_size);
        while (remaining_size > 0) {
          int task_size = std::min(remaining_size, max_size);
          output_tasks->emplace_back(new FakeTask(task_size));
          remaining_size -= task_size;
        }
        return Status::OK();
      };
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));
  TF_ASSERT_OK(ScheduleTask(30, queue.get()));
  TF_ASSERT_OK(ScheduleTask(350, queue.get()));
  // Second task should be split into a task of size 70, 2 tasks of size 100,
  // and one task of size 80.
  while (true) {
    mutex_lock l(mu);
    if (processed_batches == 4) break;
  }
}

TEST(AdaptiveSharedBatchSchedulerTest, MaxTasksPerBatch) {
  mutex mu;
  int processed_batches = 0;
  auto queue_callback =
      [&mu, &processed_batches](std::unique_ptr<Batch<FakeTask>> batch) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSadaptive_shared_batch_scheduler_testDTcc mht_13(mht_13_v, 697, "", "./tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler_test.cc", "lambda");

        ASSERT_TRUE(batch->IsClosed());
        mutex_lock l(mu);
        ++processed_batches;
      };
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;

  AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.max_batch_size = 100;
  queue_options.batch_timeout_micros = 1000000;
  queue_options.max_tasks_per_batch = 2;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  // Only one task in the batch, so batch not processed yet.
  EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  // Two tasks were added to the batch, so batch was processed.
  EXPECT_EQ(queue->NumEnqueuedTasks(), 0);
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  TF_ASSERT_OK(ScheduleTask(10, queue.get()));
  // We get 3 batches since only two tasks are allowed per batch.
  while (true) {
    mutex_lock l(mu);
    if (processed_batches == 3) break;
  }
}
}  // namespace anonymous
}  // namespace serving
}  // namespace tensorflow
