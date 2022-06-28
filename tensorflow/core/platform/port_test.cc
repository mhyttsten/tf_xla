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
class MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc() {
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

#include <condition_variable>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace port {

TEST(Port, AlignedMalloc) {
  for (size_t alignment = 1; alignment <= 1 << 20; alignment <<= 1) {
    void* p = AlignedMalloc(1, alignment);
    ASSERT_TRUE(p != nullptr) << "AlignedMalloc(1, " << alignment << ")";
    uintptr_t pval = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(pval % alignment, 0);
    AlignedFree(p);
  }
}

TEST(Port, GetCurrentCPU) {
  const int cpu = GetCurrentCPU();
#if !defined(__APPLE__)
  // GetCurrentCPU does not currently work on MacOS.
  EXPECT_GE(cpu, 0);
  EXPECT_LT(cpu, NumTotalCPUs());
#endif
}

TEST(ConditionVariable, WaitForMilliseconds_Timeout) {
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  ConditionResult result = kCond_MaybeNotified;
  time_t start = time(nullptr);
  // Condition variables are subject to spurious wakeups on some platforms,
  // so need to check for a timeout within a loop.
  while (result == kCond_MaybeNotified) {
    result = WaitForMilliseconds(&l, &cv, 3000);
  }
  EXPECT_EQ(result, kCond_Timeout);
  time_t finish = time(nullptr);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionVariable, WaitForMilliseconds_Signalled) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  mutex m;
  mutex_lock l(m);
  condition_variable cv;
  time_t start = time(nullptr);
  // Sleep for just 1 second then notify.  We have a timeout of 3 secs,
  // so the condition variable will notice the cv signal before the timeout.
  pool.Schedule([&m, &cv]() {
    Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
    mutex_lock l(m);
    cv.notify_all();
  });
  EXPECT_EQ(WaitForMilliseconds(&l, &cv, 3000), kCond_MaybeNotified);
  time_t finish = time(nullptr);
  EXPECT_LT(finish - start, 3);
}

TEST(ConditionalCriticalSections, AwaitWithDeadline_Timeout) {
  bool always_false = false;
  mutex m;
  m.lock();
  time_t start = time(nullptr);
  bool result =
      m.AwaitWithDeadline(Condition(&always_false),
                          EnvTime::NowNanos() + 3 * EnvTime::kSecondsToNanos);
  time_t finish = time(nullptr);
  m.unlock();
  EXPECT_EQ(result, false);
  EXPECT_GE(finish - start, 3);
}

TEST(ConditionalCriticalSections, AwaitWithDeadline_Woken) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool woken = false;
  mutex m;
  m.lock();
  time_t start = time(nullptr);
  // Sleep for just 1 second then set the boolean.  We have a timeout of 3
  // secs, so the mutex implementation will notice the boolean state change
  // before the timeout.
  pool.Schedule([&m, &woken]() {
    Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
    m.lock();
    woken = true;
    m.unlock();
  });
  bool result = m.AwaitWithDeadline(
      Condition(&woken), EnvTime::NowNanos() + 3 * EnvTime::kSecondsToNanos);
  time_t finish = time(nullptr);
  m.unlock();
  EXPECT_EQ(result, true);
  EXPECT_LT(finish - start, 3);
}

// Return the negation of *b.  Used as an Await() predicate.
static bool Invert(bool* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc mht_0(mht_0_v, 288, "", "./tensorflow/core/platform/port_test.cc", "Invert");
 return !*b; }

// The Value() method inverts the value of the boolean specified in
// the constructor.
class InvertClass {
 public:
  explicit InvertClass(bool* value) : value_(value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc mht_1(mht_1_v, 297, "", "./tensorflow/core/platform/port_test.cc", "InvertClass");
}
  bool Value() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSport_testDTcc mht_2(mht_2_v, 301, "", "./tensorflow/core/platform/port_test.cc", "Value");
 return !*this->value_; }

 private:
  InvertClass();
  bool* value_;
};

TEST(ConditionalCriticalSections, Await_PingPong) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool ping_pong = false;
  bool done = false;
  mutex m;
  pool.Schedule([&m, &ping_pong, &done]() {
    m.lock();
    for (int i = 0; i != 1000; i++) {
      m.Await(Condition(&ping_pong));
      ping_pong = false;
    }
    done = true;
    m.unlock();
  });
  m.lock();
  InvertClass invert(&ping_pong);
  for (int i = 0; i != 1000; i++) {
    m.Await(Condition(&Invert, &ping_pong));
    ping_pong = true;
  }
  m.Await(Condition(&done));
  m.unlock();
}

TEST(ConditionalCriticalSections, Await_PingPongMethod) {
  thread::ThreadPool pool(Env::Default(), "test", 1);
  bool ping_pong = false;
  bool done = false;
  mutex m;
  pool.Schedule([&m, &ping_pong, &done]() {
    m.lock();
    for (int i = 0; i != 1000; i++) {
      m.Await(Condition(&ping_pong));
      ping_pong = false;
    }
    done = true;
    m.unlock();
  });
  m.lock();
  InvertClass invert(&ping_pong);
  for (int i = 0; i != 1000; i++) {
    m.Await(Condition(&invert, &InvertClass::Value));
    ping_pong = true;
  }
  m.Await(Condition(&done));
  m.unlock();
}

TEST(TestCPUFeature, TestFeature) {
  // We don't know what the result should be on this platform, so just make
  // sure it's callable.
  const bool has_avx = TestCPUFeature(CPUFeature::AVX);
  LOG(INFO) << "has_avx = " << has_avx;
  const bool has_avx2 = TestCPUFeature(CPUFeature::AVX2);
  LOG(INFO) << "has_avx2 = " << has_avx2;
}

}  // namespace port
}  // namespace tensorflow
