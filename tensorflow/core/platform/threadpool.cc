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
class MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc() {
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

#include "tensorflow/core/platform/threadpool.h"

#define EIGEN_USE_THREADS

#include "absl/types/optional.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const string& name)
      : env_(env), thread_options_(thread_options), name_(name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/platform/threadpool.cc", "EigenEnvironment");
}

  EnvThread* CreateThread(std::function<void()> f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/platform/threadpool.cc", "CreateThread");

    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      port::ScopedSetRound round(FE_TONEAREST);
      if (thread_options_.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/platform/threadpool.cc", "CreateTask");

    uint64 id = 0;
    if (tracing::EventCollector::IsEnabled()) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/platform/threadpool.cc", "ExecuteTask");

    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->f();
  }
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ThreadPool");
}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ThreadPool");
}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_6(mht_6_v, 289, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ThreadPool");

  CHECK_GE(num_threads, 1);
  eigen_threadpool_.reset(new Eigen::ThreadPoolTempl<EigenEnvironment>(
      num_threads, low_latency_hint,
      EigenEnvironment(env, thread_options, "tf_" + name)));
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(underlying_threadpool_,
                                                       num_threads, allocator));
}

ThreadPool::ThreadPool(thread::ThreadPoolInterface* user_threadpool) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_7(mht_7_v, 302, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ThreadPool");

  underlying_threadpool_ = user_threadpool;
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(
      underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr));
}

ThreadPool::~ThreadPool() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_8(mht_8_v, 311, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::~ThreadPool");
}

void ThreadPool::Schedule(std::function<void()> fn) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::Schedule");

  CHECK(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_10(mht_10_v, 325, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::NumShardsUsedByFixedBlockSizeScheduling");

  if (block_size <= 0 || total <= 1 || total <= block_size ||
      NumThreads() == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(
    const int64_t block_size, const int64_t total) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_11(mht_11_v, 337, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::NumShardsUsedByTransformRangeConcurrently");

  return NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
}

void ThreadPool::ParallelFor(int64_t total,
                             const SchedulingParams& scheduling_params,
                             const std::function<void(int64_t, int64_t)>& fn) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_12(mht_12_v, 346, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ParallelFor");

  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, *scheduling_params.cost_per_unit(), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(
            total, *scheduling_params.block_size(), fn);
      }
      break;
    }
  }
}

void ThreadPool::TransformRangeConcurrently(
    const int64_t block_size, const int64_t total,
    const std::function<void(int64_t, int64_t)>& fn) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_13(mht_13_v, 369, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::TransformRangeConcurrently");

  ParallelFor(total,
              SchedulingParams(SchedulingStrategy::kFixedBlockSize,
                               absl::nullopt /* cost_per_unit */, block_size),
              fn);
}

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size,
    const std::function<void(int64_t, int64_t)>& fn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_14(mht_14_v, 383, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ParallelForFixedBlockSizeScheduling");

  const int num_shards_used =
      NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(int64_t, int64_t)> handle_range =
      [=, &handle_range, &counter, &fn](int64_t first, int64_t last) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_15(mht_15_v, 397, "", "./tensorflow/core/platform/threadpool.cc", "lambda");

        while (last - first > block_size) {
          // Find something near the midpoint which is a multiple of block size.
          const int64_t mid = first + ((last - first) / 2 + block_size - 1) /
                                          block_size * block_size;
          Schedule([=, &handle_range]() { handle_range(mid, last); });
          last = mid;
        }
        // Single block or less, execute directly.
        fn(first, last);
        counter.DecrementCount();  // The shard is done.
      };
  if (num_shards_used <= NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(0, total);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    Schedule([=, &handle_range]() { handle_range(0, total); });
  }
  counter.Wait();
}

void ThreadPool::ParallelFor(int64_t total, int64_t cost_per_unit,
                             const std::function<void(int64_t, int64_t)>& fn) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_16(mht_16_v, 425, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ParallelFor");

  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64_t)(Eigen::Index)total);
  threadpool_device_->parallelFor(
      total, Eigen::TensorOpCost(0, 0, cost_per_unit),
      [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
}

void ThreadPool::ParallelForWithWorkerId(
    int64_t total, int64_t cost_per_unit,
    const std::function<void(int64_t, int64_t, int)>& fn) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_17(mht_17_v, 438, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ParallelForWithWorkerId");

  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64_t)(Eigen::Index)total);

  threadpool_device_->parallelFor(total,
                                  Eigen::TensorOpCost(0, 0, cost_per_unit),
                                  [this, &fn](int64_t start, int64_t limit) {
                                    // ParallelFor may use the current thread to
                                    // do some work synchronously. When calling
                                    // CurrentThreadId() from outside of the
                                    // thread pool, we get -1, so we can shift
                                    // every id up by 1.
                                    int id = CurrentThreadId() + 1;
                                    fn(start, limit, id);
                                  });
}

void ThreadPool::ParallelForWithWorkerId(
    int64_t total, const SchedulingParams& scheduling_params,
    const std::function<void(int64_t, int64_t, int)>& fn) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_18(mht_18_v, 460, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ParallelForWithWorkerId");

  ParallelFor(total, scheduling_params,
              [this, &fn](int64_t start, int64_t limit) {
                // We may use the current thread to do some work synchronously.
                // When calling CurrentThreadId() from outside of the thread
                // pool, we get -1, so we can shift every id up by 1.
                int id = CurrentThreadId() + 1;
                fn(start, limit, id);
              });
}

int ThreadPool::NumThreads() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_19(mht_19_v, 474, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::NumThreads");

  return underlying_threadpool_->NumThreads();
}

int ThreadPool::CurrentThreadId() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_20(mht_20_v, 481, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::CurrentThreadId");

  return underlying_threadpool_->CurrentThreadId();
}

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start,
                                  int limit) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_21(mht_21_v, 489, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::ScheduleWithHint");

  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(
    const std::vector<std::pair<unsigned, unsigned>>& partitions) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_22(mht_22_v, 497, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::SetStealPartitions");

  // ThreadPool::SetStealPartitions is only called in the constructor of
  // RunHandlerPool::Impl, which currently instantiates ThreadPool using a
  // constructor that does not take user_threadpool. Thus we assume
  // eigen_threadpool_ is not null here.
  DCHECK(eigen_threadpool_ != nullptr);
  eigen_threadpool_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTcc mht_23(mht_23_v, 509, "", "./tensorflow/core/platform/threadpool.cc", "ThreadPool::AsEigenThreadPool");

  DCHECK(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace thread
}  // namespace tensorflow
