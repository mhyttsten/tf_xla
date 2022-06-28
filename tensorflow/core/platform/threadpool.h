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

#ifndef TENSORFLOW_CORE_PLATFORM_THREADPOOL_H_
#define TENSORFLOW_CORE_PLATFORM_THREADPOOL_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTh() {
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


#include <functional>
#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen {
class Allocator;
class ThreadPoolInterface;
struct ThreadPoolDevice;

template <typename Environment>
class ThreadPoolTempl;
}  // namespace Eigen

namespace tensorflow {
namespace thread {

struct EigenEnvironment;

class ThreadPool {
 public:
  // Scheduling strategies for ParallelFor. The strategy governs how the given
  // units of work are distributed among the available threads in the
  // threadpool.
  enum class SchedulingStrategy {
    // The Adaptive scheduling strategy adaptively chooses the shard sizes based
    // on the cost of each unit of work, and the cost model of the underlying
    // threadpool device.
    //
    // The 'cost_per_unit' is an estimate of the number of CPU cycles (or
    // nanoseconds if not CPU-bound) to complete a unit of work. Overestimating
    // creates too many shards and CPU time will be dominated by per-shard
    // overhead, such as Context creation. Underestimating may not fully make
    // use of the specified parallelism, and may also cause inefficiencies due
    // to load balancing issues and stragglers.
    kAdaptive,
    // The Fixed Block Size scheduling strategy shards the given units of work
    // into shards of fixed size. In case the total number of units is not
    // evenly divisible by 'block_size', at most one of the shards may be of
    // smaller size. The exact number of shards may be found by a call to
    // NumShardsUsedByFixedBlockSizeScheduling.
    //
    // Each shard may be executed on a different thread in parallel, depending
    // on the number of threads available in the pool. Note that when there
    // aren't enough threads in the pool to achieve full parallelism, function
    // calls will be automatically queued.
    kFixedBlockSize
  };

  // Contains additional parameters for either the Adaptive or the Fixed Block
  // Size scheduling strategy.
  class SchedulingParams {
   public:
    explicit SchedulingParams(SchedulingStrategy strategy,
                              absl::optional<int64_t> cost_per_unit,
                              absl::optional<int64_t> block_size)
        : strategy_(strategy),
          cost_per_unit_(cost_per_unit),
          block_size_(block_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTh mht_0(mht_0_v, 250, "", "./tensorflow/core/platform/threadpool.h", "SchedulingParams");
}

    SchedulingStrategy strategy() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSthreadpoolDTh mht_1(mht_1_v, 255, "", "./tensorflow/core/platform/threadpool.h", "strategy");
 return strategy_; }
    absl::optional<int64_t> cost_per_unit() const { return cost_per_unit_; }
    absl::optional<int64_t> block_size() const { return block_size_; }

   private:
    // The underlying Scheduling Strategy for which this instance contains
    // additional parameters.
    SchedulingStrategy strategy_;

    // The estimated cost per unit of work in number of CPU cycles (or
    // nanoseconds if not CPU-bound). Only applicable for Adaptive scheduling
    // strategy.
    absl::optional<int64_t> cost_per_unit_;

    // The block size of each shard. Only applicable for Fixed Block Size
    // scheduling strategy.
    absl::optional<int64_t> block_size_;
  };

  // Constructs a pool that contains "num_threads" threads with specified
  // "name". env->StartThread() is used to create individual threads with the
  // given ThreadOptions. If "low_latency_hint" is true the thread pool
  // implementation may use it as a hint that lower latency is preferred at the
  // cost of higher CPU usage, e.g. by letting one or more idle threads spin
  // wait. Conversely, if the threadpool is used to schedule high-latency
  // operations like I/O the hint should be set to false.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options,
             const std::string& name, int num_threads, bool low_latency_hint,
             Eigen::Allocator* allocator = nullptr);

  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const std::string& name, int num_threads);

  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads with the given ThreadOptions.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options,
             const std::string& name, int num_threads);

  // Constructs a pool that wraps around the thread::ThreadPoolInterface
  // instance provided by the caller. Caller retains ownership of
  // `user_threadpool` and must ensure its lifetime is longer than the
  // ThreadPool instance.
  explicit ThreadPool(thread::ThreadPoolInterface* user_threadpool);

  // Waits until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn);

  void SetStealPartitions(
      const std::vector<std::pair<unsigned, unsigned>>& partitions);

  void ScheduleWithHint(std::function<void()> fn, int start, int limit);

  // Returns the number of shards used by ParallelForFixedBlockSizeScheduling
  // with these parameters.
  int NumShardsUsedByFixedBlockSizeScheduling(const int64_t total,
                                              const int64_t block_size);

  // Returns the number of threads spawned by calling TransformRangeConcurrently
  // with these parameters.
  // Deprecated. Use NumShardsUsedByFixedBlockSizeScheduling.
  int NumShardsUsedByTransformRangeConcurrently(const int64_t block_size,
                                                const int64_t total);

  // ParallelFor shards the "total" units of work assuming each unit of work
  // having roughly "cost_per_unit" cost, in cycles. Each unit of work is
  // indexed 0, 1, ..., total - 1. Each shard contains 1 or more units of work
  // and the total cost of each shard is roughly the same.
  //
  // "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
  // if not CPU-bound) to complete a unit of work. Overestimating creates too
  // many shards and CPU time will be dominated by per-shard overhead, such as
  // Context creation. Underestimating may not fully make use of the specified
  // parallelism, and may also cause inefficiencies due to load balancing
  // issues and stragglers.
  void ParallelFor(int64_t total, int64_t cost_per_unit,
                   const std::function<void(int64_t, int64_t)>& fn);

  // Similar to ParallelFor above, but takes the specified scheduling strategy
  // into account.
  void ParallelFor(int64_t total, const SchedulingParams& scheduling_params,
                   const std::function<void(int64_t, int64_t)>& fn);

  // Same as ParallelFor with Fixed Block Size scheduling strategy.
  // Deprecated. Prefer ParallelFor with a SchedulingStrategy argument.
  void TransformRangeConcurrently(
      const int64_t block_size, const int64_t total,
      const std::function<void(int64_t, int64_t)>& fn);

  // Shards the "total" units of work. For more details, see "ParallelFor".
  //
  // The function is passed a thread_id between 0 and NumThreads() *inclusive*.
  // This is because some work can happen on the caller thread while the threads
  // in the pool are also being used.
  //
  // The caller can allocate NumThreads() + 1 separate buffers for each thread.
  // Each thread can safely write to the buffer given by its id without
  // synchronization. However, the worker fn may be called multiple times
  // sequentially with the same id.
  //
  // At most NumThreads() unique ids will actually be used, and only a few may
  // be used for small workloads. If each buffer is expensive, the buffers
  // should be stored in an array initially filled with null, and a buffer
  // should be allocated by fn the first time that the id is used.
  void ParallelForWithWorkerId(
      int64_t total, int64_t cost_per_unit,
      const std::function<void(int64_t, int64_t, int)>& fn);

  // Similar to ParallelForWithWorkerId above, but takes the specified
  // scheduling strategy into account.
  void ParallelForWithWorkerId(
      int64_t total, const SchedulingParams& scheduling_params,
      const std::function<void(int64_t, int64_t, int)>& fn);

  // Returns the number of threads in the pool.
  int NumThreads() const;

  // Returns current thread id between 0 and NumThreads() - 1, if called from a
  // thread in the pool. Returns -1 otherwise.
  int CurrentThreadId() const;

  // If ThreadPool implementation is compatible with Eigen::ThreadPoolInterface,
  // returns a non-null pointer. The caller does not own the object the returned
  // pointer points to, and should not attempt to delete.
  Eigen::ThreadPoolInterface* AsEigenThreadPool() const;

 private:
  // Divides the work represented by the range [0, total) into k shards.
  // Calls fn(i*block_size, (i+1)*block_size) from the ith shard (0 <= i < k).
  // Each shard may be executed on a different thread in parallel, depending on
  // the number of threads available in the pool.
  // When (i+1)*block_size > total, fn(i*block_size, total) is called instead.
  // Here, k = NumShardsUsedByFixedBlockSizeScheduling(total, block_size).
  // Requires 0 < block_size <= total.
  void ParallelForFixedBlockSizeScheduling(
      const int64_t total, const int64_t block_size,
      const std::function<void(int64_t, int64_t)>& fn);

  // underlying_threadpool_ is the user_threadpool if user_threadpool is
  // provided in the constructor. Otherwise it is the eigen_threadpool_.
  Eigen::ThreadPoolInterface* underlying_threadpool_;
  // eigen_threadpool_ is instantiated and owned by thread::ThreadPool if
  // user_threadpool is not in the constructor.
  std::unique_ptr<Eigen::ThreadPoolTempl<EigenEnvironment>> eigen_threadpool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
  TF_DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_THREADPOOL_H_
