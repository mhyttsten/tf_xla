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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc() {
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

#include "tensorflow/core/kernels/batching_util/periodic_function.h"

#include <memory>
#include <string>

#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {

namespace internal {

class PeriodicFunctionTestAccess {
 public:
  explicit PeriodicFunctionTestAccess(PeriodicFunction* periodic_function)
      : periodic_function_(periodic_function) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "PeriodicFunctionTestAccess");
}

  void NotifyStop() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "NotifyStop");
 periodic_function_->NotifyStop(); }

 private:
  PeriodicFunction* const periodic_function_;
};

}  // namespace internal

namespace {

using test_util::FakeClockEnv;

void StopPeriodicFunction(PeriodicFunction* periodic_function,
                          FakeClockEnv* fake_clock_env,
                          const uint64 pf_interval_micros) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "StopPeriodicFunction");

  fake_clock_env->BlockUntilThreadsAsleep(1);
  internal::PeriodicFunctionTestAccess(periodic_function).NotifyStop();
  fake_clock_env->AdvanceByMicroseconds(pf_interval_micros);
}

TEST(PeriodicFunctionTest, ObeyInterval) {
  const int64_t kPeriodMicros = 2;
  const int kCalls = 10;

  int actual_calls = 0;
  {
    FakeClockEnv fake_clock_env(Env::Default());
    PeriodicFunction::Options options;
    options.env = &fake_clock_env;
    PeriodicFunction periodic_function([&actual_calls]() { ++actual_calls; },
                                       kPeriodMicros, options);

    for (int i = 0; i < kCalls; ++i) {
      fake_clock_env.BlockUntilThreadsAsleep(1);
      fake_clock_env.AdvanceByMicroseconds(kPeriodMicros);
    }
    StopPeriodicFunction(&periodic_function, &fake_clock_env, kPeriodMicros);
  }

  // The function gets called kCalls+1 times: once at time 0, once at time
  // kPeriodMicros, once at time kPeriodMicros*2, up to once at time
  // kPeriodMicros*kCalls.
  ASSERT_EQ(actual_calls, kCalls + 1);
}

TEST(PeriodicFunctionTest, ObeyStartupDelay) {
  const int64_t kDelayMicros = 10;
  const int64_t kPeriodMicros = kDelayMicros / 10;

  int actual_calls = 0;
  {
    PeriodicFunction::Options options;
    options.startup_delay_micros = kDelayMicros;
    FakeClockEnv fake_clock_env(Env::Default());
    options.env = &fake_clock_env;
    PeriodicFunction periodic_function([&actual_calls]() { ++actual_calls; },
                                       kPeriodMicros, options);

    // Wait for the thread to start up.
    fake_clock_env.BlockUntilThreadsAsleep(1);
    // Function shouldn't have been called yet.
    EXPECT_EQ(0, actual_calls);
    // Give enough time for startup delay to expire.
    fake_clock_env.AdvanceByMicroseconds(kDelayMicros);
    StopPeriodicFunction(&periodic_function, &fake_clock_env, kDelayMicros);
  }

  // Function should have been called at least once.
  EXPECT_EQ(1, actual_calls);
}

// Test for race in calculating the first time the callback should fire.
TEST(PeriodicFunctionTest, StartupDelayRace) {
  const int64_t kDelayMicros = 10;
  const int64_t kPeriodMicros = kDelayMicros / 10;

  mutex mu;
  int counter = 0;
  std::unique_ptr<Notification> listener(new Notification);

  FakeClockEnv fake_clock_env(Env::Default());
  PeriodicFunction::Options options;
  options.env = &fake_clock_env;
  options.startup_delay_micros = kDelayMicros;
  PeriodicFunction periodic_function(
      [&mu, &counter, &listener]() {
        mutex_lock l(mu);
        counter++;
        listener->Notify();
      },
      kPeriodMicros, options);

  fake_clock_env.BlockUntilThreadsAsleep(1);
  fake_clock_env.AdvanceByMicroseconds(kDelayMicros);
  listener->WaitForNotification();
  {
    mutex_lock l(mu);
    EXPECT_EQ(1, counter);
    // A notification can only be notified once.
    listener.reset(new Notification);
  }
  fake_clock_env.BlockUntilThreadsAsleep(1);
  fake_clock_env.AdvanceByMicroseconds(kPeriodMicros);
  listener->WaitForNotification();
  {
    mutex_lock l(mu);
    EXPECT_EQ(2, counter);
  }
  StopPeriodicFunction(&periodic_function, &fake_clock_env, kPeriodMicros);
}

// If this test hangs forever, its probably a deadlock caused by setting the
// PeriodicFunction's interval to 0ms.
TEST(PeriodicFunctionTest, MinInterval) {
  PeriodicFunction periodic_function(
      []() { Env::Default()->SleepForMicroseconds(20 * 1000); }, 0);
}

class PeriodicFunctionWithFakeClockEnvTest : public ::testing::Test {
 protected:
  const int64_t kPeriodMicros = 50;
  PeriodicFunctionWithFakeClockEnvTest()
      : fake_clock_env_(Env::Default()),
        counter_(0),
        pf_(
            [this]() {
              mutex_lock l(counter_mu_);
              ++counter_;
            },
            kPeriodMicros, GetPeriodicFunctionOptions()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_3(mht_3_v, 341, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "PeriodicFunctionWithFakeClockEnvTest");
}

  PeriodicFunction::Options GetPeriodicFunctionOptions() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_4(mht_4_v, 346, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "GetPeriodicFunctionOptions");

    PeriodicFunction::Options options;
    options.thread_name_prefix = "ignore";
    options.env = &fake_clock_env_;
    return options;
  }

  void SetUp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_5(mht_5_v, 356, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "SetUp");

    // Note: counter_ gets initially incremented at time 0.
    ASSERT_TRUE(AwaitCount(1));
  }

  void TearDown() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_6(mht_6_v, 364, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "TearDown");

    StopPeriodicFunction(&pf_, &fake_clock_env_, kPeriodMicros);
  }

  // The FakeClockEnv tests below advance simulated time and then expect the
  // PeriodicFunction thread to run its function. This method helps the tests
  // wait for the thread to execute, and then check the count matches the
  // expectation.
  bool AwaitCount(int expected_counter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_function_testDTcc mht_7(mht_7_v, 375, "", "./tensorflow/core/kernels/batching_util/periodic_function_test.cc", "AwaitCount");

    fake_clock_env_.BlockUntilThreadsAsleep(1);
    {
      mutex_lock lock(counter_mu_);
      return counter_ == expected_counter;
    }
  }

  FakeClockEnv fake_clock_env_;
  mutex counter_mu_;
  int counter_;
  PeriodicFunction pf_;
};

TEST_F(PeriodicFunctionWithFakeClockEnvTest, FasterThanRealTime) {
  fake_clock_env_.AdvanceByMicroseconds(kPeriodMicros / 2);
  for (int i = 2; i < 7; ++i) {
    fake_clock_env_.AdvanceByMicroseconds(
        kPeriodMicros);  // advance past a tick
    EXPECT_TRUE(AwaitCount(i));
  }
}

TEST_F(PeriodicFunctionWithFakeClockEnvTest, SlowerThanRealTime) {
  Env::Default()->SleepForMicroseconds(
      125 * 1000);  // wait for any unexpected breakage
  EXPECT_TRUE(AwaitCount(1));
}

TEST(PeriodicFunctionDeathTest, BadInterval) {
  EXPECT_DEBUG_DEATH(PeriodicFunction periodic_function([]() {}, -1),
                     ".* should be >= 0");

  EXPECT_DEBUG_DEATH(PeriodicFunction periodic_function(
                         []() {}, -1, PeriodicFunction::Options()),
                     ".* should be >= 0");
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
