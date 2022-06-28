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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc() {
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

#include "tensorflow/core/common_runtime/allocator_retry.h"

#include <vector>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class FakeAllocator {
 public:
  FakeAllocator(size_t cap, int millis_to_wait)
      : memory_capacity_(cap), millis_to_wait_(millis_to_wait) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "FakeAllocator");
}

  // Allocate just keeps track of the number of outstanding allocations,
  // not their sizes.  Assume a constant size for each.
  void* AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "AllocateRaw");

    return retry_.AllocateRaw(
        [this](size_t a, size_t nb, bool v) {
          mutex_lock l(mu_);
          if (memory_capacity_ > 0) {
            --memory_capacity_;
            return good_ptr_;
          } else {
            return static_cast<void*>(nullptr);
          }
        },
        millis_to_wait_, alignment, num_bytes);
  }

  void DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "DeallocateRaw");

    mutex_lock l(mu_);
    ++memory_capacity_;
    retry_.NotifyDealloc();
  }

 private:
  AllocatorRetry retry_;
  void* good_ptr_ = reinterpret_cast<void*>(0xdeadbeef);
  mutex mu_;
  size_t memory_capacity_ TF_GUARDED_BY(mu_);
  int millis_to_wait_;
};

// GPUAllocatorRetry is a mechanism to deal with race conditions which
// are inevitable in the TensorFlow runtime where parallel Nodes can
// execute in any order.  Properly testing this feature would use real
// multi-threaded race conditions, but that leads to flaky tests as
// the expected outcome fails to occur with low but non-zero
// probability.  To make these tests reliable we simulate real race
// conditions by forcing parallel threads to take turns in the
// interesting part of their interaction with the allocator.  This
// class is the mechanism that imposes turn taking.
class AlternatingBarrier {
 public:
  explicit AlternatingBarrier(int num_users)
      : num_users_(num_users), next_turn_(0), done_(num_users, false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "AlternatingBarrier");
}

  void WaitTurn(int user_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "WaitTurn");

    mutex_lock l(mu_);
    int wait_cycles = 0;
    // A user is allowed to proceed out of turn if it waits too long.
    while (next_turn_ != user_index && wait_cycles++ < 10) {
      cv_.wait_for(l, std::chrono::milliseconds(1));
    }
    if (next_turn_ == user_index) {
      IncrementTurn();
      cv_.notify_all();
    }
  }

  // When a user quits, stop reserving it a turn.
  void Done(int user_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "Done");

    mutex_lock l(mu_);
    done_[user_index] = true;
    if (next_turn_ == user_index) {
      IncrementTurn();
      cv_.notify_all();
    }
  }

 private:
  void IncrementTurn() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_6(mht_6_v, 290, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "IncrementTurn");

    int skipped = 0;
    while (skipped < num_users_) {
      next_turn_ = (next_turn_ + 1) % num_users_;
      if (!done_[next_turn_]) return;
      ++skipped;
    }
  }

  mutex mu_;
  condition_variable cv_;
  int num_users_;
  int next_turn_ TF_GUARDED_BY(mu_);
  std::vector<bool> done_ TF_GUARDED_BY(mu_);
};

class GPUAllocatorRetryTest : public ::testing::Test {
 protected:
  GPUAllocatorRetryTest() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_7(mht_7_v, 311, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "GPUAllocatorRetryTest");
}

  void LaunchConsumerThreads(int num_consumers, int cap_needed) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "LaunchConsumerThreads");

    barrier_.reset(new AlternatingBarrier(num_consumers));
    consumer_count_.resize(num_consumers, 0);
    for (int i = 0; i < num_consumers; ++i) {
      consumers_.push_back(Env::Default()->StartThread(
          ThreadOptions(), "anon_thread", [this, i, cap_needed]() {
            do {
              void* ptr = nullptr;
              for (int j = 0; j < cap_needed; ++j) {
                barrier_->WaitTurn(i);
                ptr = alloc_->AllocateRaw(16, 1);
                if (ptr == nullptr) {
                  mutex_lock l(mu_);
                  has_failed_ = true;
                  barrier_->Done(i);
                  return;
                }
              }
              ++consumer_count_[i];
              for (int j = 0; j < cap_needed; ++j) {
                barrier_->WaitTurn(i);
                alloc_->DeallocateRaw(ptr);
              }
            } while (!notifier_.HasBeenNotified());
            barrier_->Done(i);
          }));
    }
  }

  // Wait up to wait_micros microseconds for has_failed_ to equal expected,
  // then terminate all threads.
  void JoinConsumerThreads(bool expected, int wait_micros) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_allocator_retry_testDTcc mht_9(mht_9_v, 350, "", "./tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc", "JoinConsumerThreads");

    while (wait_micros > 0) {
      {
        mutex_lock l(mu_);
        if (has_failed_ == expected) break;
      }
      int interval_micros = std::min(1000, wait_micros);
      Env::Default()->SleepForMicroseconds(interval_micros);
      wait_micros -= interval_micros;
    }
    notifier_.Notify();
    for (auto c : consumers_) {
      // Blocks until thread terminates.
      delete c;
    }
  }

  std::unique_ptr<FakeAllocator> alloc_;
  std::unique_ptr<AlternatingBarrier> barrier_;
  std::vector<Thread*> consumers_;
  std::vector<int> consumer_count_;
  Notification notifier_;
  mutex mu_;
  bool has_failed_ TF_GUARDED_BY(mu_) = false;
  int count_ TF_GUARDED_BY(mu_) = 0;
};

// Verifies correct retrying when memory is slightly overcommitted but
// we allow retry.
TEST_F(GPUAllocatorRetryTest, RetrySuccess) {
  // Support up to 2 allocations simultaneously, waits up to 1000 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 1000));
  // Launch 3 consumers, each of whom needs 1 unit at a time.
  LaunchConsumerThreads(3, 1);
  // This should be enough time for each consumer to be satisfied many times.
  Env::Default()->SleepForMicroseconds(50000);
  JoinConsumerThreads(false, 0);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_FALSE(has_failed_);
  }
  EXPECT_GT(consumer_count_[0], 0);
  EXPECT_GT(consumer_count_[1], 0);
  EXPECT_GT(consumer_count_[2], 0);
}

// Verifies OutOfMemory failure when memory is slightly overcommitted
// and retry is not allowed.  Note that this test will fail, i.e. no
// memory alloc failure will be detected, if it is run in a context that
// does not permit real multi-threaded execution.
TEST_F(GPUAllocatorRetryTest, NoRetryFail) {
  // Support up to 2 allocations simultaneously, waits up to 0 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 0));
  // Launch 3 consumers, each of whom needs 1 unit at a time.
  LaunchConsumerThreads(3, 1);
  Env::Default()->SleepForMicroseconds(50000);
  // Will wait up to 10 seconds for proper race condition to occur, resulting
  // in failure.
  JoinConsumerThreads(true, 10000000);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_TRUE(has_failed_);
  }
}

// Verifies OutOfMemory failure when retry is allowed but memory capacity
// is too low even for retry.
TEST_F(GPUAllocatorRetryTest, RetryInsufficientFail) {
  // Support up to 2 allocations simultaneously, waits up to 1000 msec for
  // a chance to alloc.
  alloc_.reset(new FakeAllocator(2, 1000));
  // Launch 3 consumers, each of whom needs 2 units at a time.  We expect
  // deadlock where 2 consumers each hold 1 unit, and timeout trying to
  // get the second.
  LaunchConsumerThreads(3, 2);
  Env::Default()->SleepForMicroseconds(50000);
  // We're forcing a race condition, so this will fail quickly, but
  // give it 10 seconds anyway.
  JoinConsumerThreads(true, 10000000);
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Consumer " << i << " is " << consumer_count_[i];
  }
  {
    mutex_lock l(mu_);
    EXPECT_TRUE(has_failed_);
  }
}

}  // namespace
}  // namespace tensorflow
