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
class MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc() {
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

#include "tensorflow/core/lib/core/threadpool.h"

#include <atomic>

#include "absl/synchronization/barrier.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace thread {

static const int kNumThreads = 30;

TEST(ThreadPool, Empty) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    ThreadPool pool(Env::Default(), "test", num_threads);
  }
}

TEST(ThreadPool, DoWork) {
  Context outer_context(ContextKind::kThread);
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    {
      ThreadPool pool(Env::Default(), "test", num_threads);
      for (int i = 0; i < kWorkItems; i++) {
        pool.Schedule([&outer_context, &work, i]() {
          Context inner_context(ContextKind::kThread);
          ASSERT_EQ(outer_context, inner_context);
          ASSERT_FALSE(work[i].exchange(true));
        });
      }
    }
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

void RunWithFixedBlockSize(int64_t block_size, int64_t total,
                           ThreadPool* threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_0(mht_0_v, 236, "", "./tensorflow/core/lib/core/threadpool_test.cc", "RunWithFixedBlockSize");

  mutex mu;
  int64_t num_shards = 0;
  int64_t num_done_work = 0;
  std::vector<std::atomic<bool>> work(total);
  for (int i = 0; i < total; i++) {
    work[i] = false;
  }
  threads->ParallelFor(
      total,
      ThreadPool::SchedulingParams(
          ThreadPool::SchedulingStrategy::kFixedBlockSize /* strategy */,
          absl::nullopt /* cost_per_unit */, block_size /* block_size */),
      [=, &mu, &num_shards, &num_done_work, &work](int64_t start, int64_t end) {
        VLOG(1) << "Shard [" << start << "," << end << ")";
        EXPECT_GE(start, 0);
        EXPECT_LE(end, total);
        mutex_lock l(mu);
        ++num_shards;
        for (; start < end; ++start) {
          EXPECT_FALSE(work[start].exchange(true));  // No duplicate
          ++num_done_work;
        }
      });
  EXPECT_EQ(num_done_work, total);
  for (int i = 0; i < total; i++) {
    ASSERT_TRUE(work[i]);
  }
  const int64_t num_workers = (total + block_size - 1) / block_size;
  if (num_workers < threads->NumThreads()) {
    // If the intention is to limit the parallelism explicitly, we'd
    // better honor it. Ideally, even if per_thread_max_parallelism >
    // num_workers, we should expect that Shard() implementation do
    // not over-shard. Unfortunately, ThreadPoolDevice::parallelFor
    // tends to over-shard.
    EXPECT_LE(num_shards, 1 + num_workers);
  }
}

// Adapted from work_sharder_test.cc
TEST(ThreadPoolTest, ParallelForFixedBlockSizeScheduling) {
  ThreadPool threads(Env::Default(), "test", 16);
  for (auto block_size : {1, 7, 10, 64, 100, 256, 1000, 9999}) {
    for (auto diff : {0, 1, 11, 102, 1003, 10005, 1000007}) {
      const int64_t total = block_size + diff;
      RunWithFixedBlockSize(block_size, total, &threads);
    }
  }
}

void RunWithFixedBlockSizeTransformRangeConcurrently(int64_t block_size,
                                                     int64_t total,
                                                     ThreadPool* threads) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_1(mht_1_v, 291, "", "./tensorflow/core/lib/core/threadpool_test.cc", "RunWithFixedBlockSizeTransformRangeConcurrently");

  mutex mu;
  int64_t num_shards = 0;
  int64_t num_done_work = 0;
  std::vector<std::atomic<bool>> work(total);
  for (int i = 0; i < total; i++) {
    work[i] = false;
  }
  threads->TransformRangeConcurrently(
      block_size, total,
      [=, &mu, &num_shards, &num_done_work, &work](int64_t start, int64_t end) {
        VLOG(1) << "Shard [" << start << "," << end << ")";
        EXPECT_GE(start, 0);
        EXPECT_LE(end, total);
        mutex_lock l(mu);
        ++num_shards;
        for (; start < end; ++start) {
          EXPECT_FALSE(work[start].exchange(true));  // No duplicate
          ++num_done_work;
        }
      });
  EXPECT_EQ(num_done_work, total);
  for (int i = 0; i < total; i++) {
    ASSERT_TRUE(work[i]);
  }
  const int64_t num_workers = (total + block_size - 1) / block_size;
  if (num_workers < threads->NumThreads()) {
    // If the intention is to limit the parallelism explicitly, we'd
    // better honor it. Ideally, even if per_thread_max_parallelism >
    // num_workers, we should expect that Shard() implementation do
    // not over-shard. Unfortunately, ThreadPoolDevice::parallelFor
    // tends to over-shard.
    EXPECT_LE(num_shards, 1 + num_workers);
  }
}

// Adapted from work_sharder_test.cc
TEST(ThreadPoolTest, TransformRangeConcurrently) {
  ThreadPool threads(Env::Default(), "test", 16);
  for (auto block_size : {1, 7, 10, 64, 100, 256, 1000, 9999}) {
    for (auto diff : {0, 1, 11, 102, 1003, 10005, 1000007}) {
      const int64_t total = block_size + diff;
      RunWithFixedBlockSizeTransformRangeConcurrently(block_size, total,
                                                      &threads);
    }
  }
}

TEST(ThreadPoolTest, NumShardsUsedByFixedBlockSizeScheduling) {
  ThreadPool threads(Env::Default(), "test", 16);

  EXPECT_EQ(1, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   3 /* total */, 3 /* block_size */));
  EXPECT_EQ(2, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   4 /* total */, 3 /* block_size */));
  EXPECT_EQ(2, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   5 /* total */, 3 /* block_size */));
  EXPECT_EQ(2, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   6 /* total */, 3 /* block_size */));
  EXPECT_EQ(3, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   7 /* total */, 3 /* block_size */));
  EXPECT_EQ(7, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   7 /* total */, 1 /* block_size */));
  EXPECT_EQ(1, threads.NumShardsUsedByFixedBlockSizeScheduling(
                   7 /* total */, 0 /* block_size */));
}

TEST(ThreadPoolTest, NumShardsUsedByTransformRangeConcurrently) {
  ThreadPool threads(Env::Default(), "test", 16);

  EXPECT_EQ(1, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 3 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 4 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 5 /* total */));
  EXPECT_EQ(2, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 6 /* total */));
  EXPECT_EQ(3, threads.NumShardsUsedByTransformRangeConcurrently(
                   3 /* block_size */, 7 /* total */));
  EXPECT_EQ(7, threads.NumShardsUsedByTransformRangeConcurrently(
                   1 /* block_size */, 7 /* total */));
  EXPECT_EQ(1, threads.NumShardsUsedByTransformRangeConcurrently(
                   0 /* block_size */, 7 /* total */));
}

void RunFixedBlockSizeShardingWithWorkerId(int64_t block_size, int64_t total,
                                           ThreadPool* threads) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_2(mht_2_v, 381, "", "./tensorflow/core/lib/core/threadpool_test.cc", "RunFixedBlockSizeShardingWithWorkerId");

  mutex mu;
  int64_t num_done_work = 0;
  std::vector<std::atomic<bool>> work(total);
  for (int i = 0; i < total; i++) {
    work[i] = false;
  }
  const int64_t num_threads = threads->NumThreads();
  std::vector<std::atomic<bool>> threads_running(num_threads + 1);
  for (int i = 0; i < num_threads + 1; i++) {
    threads_running[i] = false;
  }

  threads->ParallelForWithWorkerId(
      total,
      ThreadPool::SchedulingParams(
          ThreadPool::SchedulingStrategy::kFixedBlockSize /* strategy */,
          absl::nullopt /* cost_per_unit */, block_size /* block_size */),
      [=, &mu, &num_done_work, &work, &threads_running](int64_t start,
                                                        int64_t end, int id) {
        VLOG(1) << "Shard [" << start << "," << end << ")";
        EXPECT_GE(start, 0);
        EXPECT_LE(end, total);

        // Store true for the current thread, and assert that another thread
        // is not running with the same id.
        EXPECT_GE(id, 0);
        EXPECT_LE(id, num_threads);
        EXPECT_FALSE(threads_running[id].exchange(true));

        mutex_lock l(mu);
        for (; start < end; ++start) {
          EXPECT_FALSE(work[start].exchange(true));  // No duplicate
          ++num_done_work;
        }
        EXPECT_TRUE(threads_running[id].exchange(false));
      });

  EXPECT_EQ(num_done_work, total);
  for (int i = 0; i < total; i++) {
    EXPECT_TRUE(work[i]);
  }
}

TEST(ThreadPoolTest, ParallelForFixedBlockSizeSchedulingWithWorkerId) {
  for (int32_t num_threads : {1, 2, 3, 9, 16, 31}) {
    ThreadPool threads(Env::Default(), "test", num_threads);
    for (int64_t block_size : {1, 7, 10, 64, 100, 256, 1000}) {
      for (int64_t diff : {0, 1, 11, 102, 1003}) {
        const int64_t total = block_size + diff;
        RunFixedBlockSizeShardingWithWorkerId(block_size, total, &threads);
      }
    }
  }
}

TEST(ThreadPool, ParallelFor) {
  Context outer_context(ContextKind::kThread);
  // Make ParallelFor use as many threads as possible.
  int64_t kHugeCost = 1 << 30;
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    ThreadPool pool(Env::Default(), "test", num_threads);
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    pool.ParallelFor(kWorkItems, kHugeCost,
                     [&outer_context, &work](int64_t begin, int64_t end) {
                       Context inner_context(ContextKind::kThread);
                       ASSERT_EQ(outer_context, inner_context);
                       for (int64_t i = begin; i < end; ++i) {
                         ASSERT_FALSE(work[i].exchange(true));
                       }
                     });
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

TEST(ThreadPool, ParallelForWithAdaptiveSchedulingStrategy) {
  Context outer_context(ContextKind::kThread);
  // Make ParallelFor use as many threads as possible.
  int64_t kHugeCost = 1 << 30;
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    ThreadPool pool(Env::Default(), "test", num_threads);
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    pool.ParallelFor(
        kWorkItems,
        ThreadPool::SchedulingParams(
            ThreadPool::SchedulingStrategy::kAdaptive /* strategy */,
            kHugeCost /* cost_per_unit */, absl::nullopt /* block_size */),
        [&outer_context, &work](int64_t begin, int64_t end) {
          Context inner_context(ContextKind::kThread);
          ASSERT_EQ(outer_context, inner_context);
          for (int64_t i = begin; i < end; ++i) {
            ASSERT_FALSE(work[i].exchange(true));
          }
        });
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

TEST(ThreadPool, ParallelForWithWorkerId) {
  // Make ParallelForWithWorkerId use as many threads as possible.
  int64_t kHugeCost = 1 << 30;
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    fprintf(stderr, "Testing with %d threads\n", num_threads);
    const int kWorkItems = 15;
    std::atomic<bool> work[kWorkItems];
    ThreadPool pool(Env::Default(), "test", num_threads);
    for (int i = 0; i < kWorkItems; i++) {
      work[i] = false;
    }
    std::atomic<bool> threads_running[kNumThreads + 1];
    for (int i = 0; i < num_threads + 1; i++) {
      threads_running[i] = false;
    }
    pool.ParallelForWithWorkerId(
        kWorkItems, kHugeCost,
        [&threads_running, &work](int64_t begin, int64_t end, int64_t id) {
          // Store true for the current thread, and assert that another thread
          // is not running with the same id.
          ASSERT_LE(0, id);
          ASSERT_LE(id, kNumThreads);
          ASSERT_FALSE(threads_running[id].exchange(true));
          for (int64_t i = begin; i < end; ++i) {
            ASSERT_FALSE(work[i].exchange(true));
          }
          ASSERT_TRUE(threads_running[id].exchange(false));
          threads_running[id] = false;
        });
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
    for (int i = 0; i < num_threads + 1; i++) {
      ASSERT_FALSE(threads_running[i]);
    }
  }
}

TEST(ThreadPool, Parallelism) {
  // Test that if we have N threads and schedule N tasks,
  // all tasks will be scheduled at the same time.
  // Failure mode for this test will be episodic timeouts (does not terminate).
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  for (int iter = 0; iter < 2000; iter++) {
    absl::Barrier barrier(kNumThreads);
    absl::BlockingCounter counter(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      pool.Schedule([&]() {
        barrier.Block();
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
}

static void BM_Sequential(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_3(mht_3_v, 552, "", "./tensorflow/core/lib/core/threadpool_test.cc", "BM_Sequential");

  for (auto s : state) {
    state.PauseTiming();
    ThreadPool pool(Env::Default(), "test", kNumThreads);
    // Decrement count sequentially until 0.
    int count = state.range(0);
    mutex done_lock;
    bool done_flag = false;
    std::function<void()> work = [&pool, &count, &done_lock, &done_flag,
                                  &work]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_4(mht_4_v, 564, "", "./tensorflow/core/lib/core/threadpool_test.cc", "lambda");

      if (count--) {
        pool.Schedule(work);
      } else {
        mutex_lock l(done_lock);
        done_flag = true;
      }
    };

    state.ResumeTiming();
    work();
    mutex_lock l(done_lock);
    done_lock.Await(Condition(&done_flag));
  }
}
BENCHMARK(BM_Sequential)->Arg(200)->Arg(300);

static void BM_Parallel(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_5(mht_5_v, 584, "", "./tensorflow/core/lib/core/threadpool_test.cc", "BM_Parallel");

  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(state.max_iterations);
  mutex done_lock;
  bool done_flag = false;

  for (auto s : state) {
    pool.Schedule([&count, &done_lock, &done_flag]() {
      if (count.fetch_sub(1) == 1) {
        mutex_lock l(done_lock);
        done_flag = true;
      }
    });
  }
  mutex_lock l(done_lock);
  done_lock.Await(Condition(&done_flag));
}
BENCHMARK(BM_Parallel);

static void BM_ParallelFor(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSthreadpool_testDTcc mht_6(mht_6_v, 607, "", "./tensorflow/core/lib/core/threadpool_test.cc", "BM_ParallelFor");

  int total = state.range(0);
  int cost_per_unit = state.range(1);
  ThreadPool pool(Env::Default(), "test", kNumThreads);
  // Decrement count concurrently until 0.
  std::atomic_int_fast32_t count(state.max_iterations);
  mutex done_lock;
  bool done_flag = false;

  for (auto s : state) {
    pool.ParallelFor(
        total, cost_per_unit,
        [&count, &done_lock, &done_flag](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            if (count.fetch_sub(1) == 1) {
              mutex_lock l(done_lock);
              done_flag = true;
            }
          }
        });

    mutex_lock l(done_lock);
    done_lock.Await(Condition(&done_flag));
  }
}
BENCHMARK(BM_ParallelFor)
    ->ArgPair(1 << 10, 1)
    ->ArgPair(1 << 20, 1)
    ->ArgPair(1 << 10, 1 << 10)
    ->ArgPair(1 << 20, 1 << 10)
    ->ArgPair(1 << 10, 1 << 20)
    ->ArgPair(1 << 20, 1 << 20)
    ->ArgPair(1 << 10, 1 << 30)
    ->ArgPair(1 << 20, 1 << 30);

}  // namespace thread
}  // namespace tensorflow
