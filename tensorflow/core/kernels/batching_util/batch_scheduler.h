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

// Abstractions for processing small tasks in a batched fashion, to reduce
// processing times and costs that can be amortized across multiple tasks.
//
// The core class is BatchScheduler, which groups tasks into batches.
//
// BatchScheduler encapsulates logic for aggregating multiple tasks into a
// batch, and kicking off processing of a batch on a thread pool it manages.
//
// This file defines an abstract BatchScheduler class.

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh() {
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


#include <stddef.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace serving {

// The abstract superclass for a unit of work to be done as part of a batch.
//
// An implementing subclass typically contains (or points to):
//  (a) input data;
//  (b) a thread-safe completion signal (e.g. a Notification);
//  (c) a place to store the outcome (success, or some error), upon completion;
//  (d) a place to store the output data, upon success.
//
// Items (b), (c) and (d) are typically non-owned pointers to data homed
// elsewhere, because a task's ownership gets transferred to a BatchScheduler
// (see below) and it may be deleted as soon as it is done executing.
class BatchTask {
 public:
  virtual ~BatchTask() = default;

  // Returns the size of the task, in terms of how much it contributes to the
  // size of a batch. (A batch's size is the sum of its task sizes.)
  virtual size_t size() const = 0;
};

// A thread-safe collection of BatchTasks, to be executed together in some
// fashion.
//
// At a given time, a batch is either "open" or "closed": an open batch can
// accept new tasks; a closed one cannot. A batch is monotonic: initially it is
// open and tasks can be added to it; then it is closed and its set of tasks
// remains fixed for the remainder of its life. A closed batch cannot be re-
// opened. Tasks can never be removed from a batch.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class Batch {
 public:
  Batch();
  explicit Batch(uint64 traceme_context_id);
  virtual ~Batch();  // Blocks until the batch is closed.

  // Appends 'task' to the batch. After calling AddTask(), the newly-added task
  // can be accessed via task(num_tasks()-1) or mutable_task(num_tasks()-1).
  // Dies if the batch is closed.
  void AddTask(std::unique_ptr<TaskType> task);

  // Removes the most recently added task. Returns nullptr if the batch is
  // empty.
  std::unique_ptr<TaskType> RemoveTask();

  // Caller takes ownership of returned tasks.
  // Must be called after a batch is closed.
  std::vector<std::unique_ptr<TaskType>> RemoveAllTasks();

  // Returns the number of tasks in the batch.
  int num_tasks() const;

  // Returns true iff the batch contains 0 tasks.
  bool empty() const;

  // Returns a reference to the ith task (in terms of insertion order).
  const TaskType& task(int i) const;

  // Returns a pointer to the ith task (in terms of insertion order).
  //
  // Caller doesn't take ownership.
  TaskType* mutable_task(int i);

  // Returns the sum of the task sizes.
  size_t size() const;

  // Returns true iff the batch is currently closed.
  bool IsClosed() const;

  // Blocks until the batch is closed.
  void WaitUntilClosed() const;

  // Marks the batch as closed. Dies if called more than once.
  void Close();

  // Returns the TraceMe context id of this batch.
  uint64 traceme_context_id() const;

 private:
  mutable mutex mu_;

  // The tasks in the batch.
  std::vector<std::unique_ptr<TaskType>> tasks_ TF_GUARDED_BY(mu_);

  // The sum of the sizes of the tasks in 'tasks_'.
  size_t size_ TF_GUARDED_BY(mu_) = 0;

  std::atomic<bool> empty_ TF_GUARDED_BY(mu_){true};

  // Whether the batch has been closed.
  Notification closed_;

  // The TracMe context id.
  const uint64 traceme_context_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(Batch);
};

// An abstract batch scheduler class. Collects individual tasks into batches,
// and processes each batch on a pool of "batch threads" that it manages. The
// actual logic for processing a batch is accomplished via a callback.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class BatchScheduler {
 public:
  virtual ~BatchScheduler() = default;

  // Submits a task to be processed as part of a batch.
  //
  // Ownership of '*task' is transferred to the callee iff the method returns
  // Status::OK. In that case, '*task' is left as nullptr. Otherwise, '*task' is
  // left as-is.
  //
  // If no batch processing capacity is available to process this task at the
  // present time, and any task queue maintained by the implementing subclass is
  // full, this method returns an UNAVAILABLE error code. The client may retry
  // later.
  //
  // Other problems, such as the task size being larger than the maximum batch
  // size, yield other, permanent error types.
  //
  // In all cases, this method returns "quickly" without blocking for any
  // substantial amount of time. If the method returns Status::OK, the task is
  // processed asynchronously, and any errors that occur during the processing
  // of the batch that includes the task can be reported to 'task'.
  virtual Status Schedule(std::unique_ptr<TaskType>* task) = 0;

  // Returns the number of tasks that have been scheduled (i.e. accepted by
  // Schedule()), but have yet to be handed to a thread for execution as part of
  // a batch. Note that this returns the number of tasks, not the aggregate task
  // size (so if there is one task of size 3 and one task of size 5, this method
  // returns 2 rather than 8).
  virtual size_t NumEnqueuedTasks() const = 0;

  // Returns a guaranteed number of size 1 tasks that can be Schedule()d without
  // getting an UNAVAILABLE error. In a typical implementation, returns the
  // available space on a queue.
  //
  // There are two important caveats:
  //  1. The guarantee does not extend to varying-size tasks due to possible
  //     internal fragmentation of batches.
  //  2. The guarantee only holds in a single-thread environment or critical
  //     section, i.e. if an intervening thread cannot call Schedule().
  //
  // This method is useful for monitoring, or for guaranteeing a future slot in
  // the schedule (but being mindful about the caveats listed above).
  virtual size_t SchedulingCapacity() const = 0;

  // Returns the maximum allowed size of tasks submitted to the scheduler. (This
  // is typically equal to a configured maximum batch size.)
  virtual size_t max_task_size() const = 0;
};

//////////
// Implementation details follow. API users need not read.

template <typename TaskType>
Batch<TaskType>::Batch() : Batch(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_0(mht_0_v, 377, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::Batch");
}

template <typename TaskType>
Batch<TaskType>::Batch(uint64 traceme_context_id)
    : traceme_context_id_(traceme_context_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_1(mht_1_v, 384, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::Batch");
}

template <typename TaskType>
Batch<TaskType>::~Batch() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_2(mht_2_v, 390, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::~Batch");

  WaitUntilClosed();
}

template <typename TaskType>
void Batch<TaskType>::AddTask(std::unique_ptr<TaskType> task) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_3(mht_3_v, 398, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::AddTask");

  DCHECK(!IsClosed());
  {
    mutex_lock l(mu_);
    size_ += task->size();
    tasks_.push_back(std::move(task));
    empty_.store(false);
  }
}

template <typename TaskType>
std::vector<std::unique_ptr<TaskType>> Batch<TaskType>::RemoveAllTasks() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_4(mht_4_v, 412, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::RemoveAllTasks");

  DCHECK(IsClosed());
  {
    mutex_lock l(mu_);
    size_ = 0;
    empty_.store(true);
    std::vector<std::unique_ptr<TaskType>> tasks_to_return;

    // Swapping vector takes constant time.
    tasks_to_return.swap(tasks_);
    return std::move(tasks_to_return);
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> Batch<TaskType>::RemoveTask() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_5(mht_5_v, 430, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::RemoveTask");

  {
    mutex_lock l(mu_);
    if (tasks_.empty()) {
      return nullptr;
    }
    std::unique_ptr<TaskType> task = std::move(tasks_.back());
    size_ -= task->size();
    tasks_.pop_back();
    if (tasks_.empty()) {
      empty_.store(true);
    }
    return task;
  }
}

template <typename TaskType>
int Batch<TaskType>::num_tasks() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_6(mht_6_v, 450, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::num_tasks");

  {
    mutex_lock l(mu_);
    return tasks_.size();
  }
}

template <typename TaskType>
bool Batch<TaskType>::empty() const TF_NO_THREAD_SAFETY_ANALYSIS {
  // tracer is added to zoom in about this method.
  // TODO(b/160249203): Remove tracer after evaluating a change to reduce
  // lock contention and cpu usage (which is observed in profiler and
  // very data-driven).
  tensorflow::profiler::TraceMe tracer("BatchTask::empty");
  return empty_.load();
}

template <typename TaskType>
const TaskType& Batch<TaskType>::task(int i) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_7(mht_7_v, 471, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::task");

  DCHECK_GE(i, 0);
  {
    mutex_lock l(mu_);
    DCHECK_LT(i, tasks_.size());
    return *tasks_[i].get();
  }
}

template <typename TaskType>
TaskType* Batch<TaskType>::mutable_task(int i) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_8(mht_8_v, 484, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::mutable_task");

  DCHECK_GE(i, 0);
  {
    mutex_lock l(mu_);
    DCHECK_LT(i, tasks_.size());
    return tasks_[i].get();
  }
}

template <typename TaskType>
size_t Batch<TaskType>::size() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_9(mht_9_v, 497, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::size");

  {
    mutex_lock l(mu_);
    return size_;
  }
}

template <typename TaskType>
bool Batch<TaskType>::IsClosed() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_10(mht_10_v, 508, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::IsClosed");

  return const_cast<Notification*>(&closed_)->HasBeenNotified();
}

template <typename TaskType>
void Batch<TaskType>::WaitUntilClosed() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_11(mht_11_v, 516, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::WaitUntilClosed");

  const_cast<Notification*>(&closed_)->WaitForNotification();
}

template <typename TaskType>
void Batch<TaskType>::Close() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_12(mht_12_v, 524, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::Close");

  closed_.Notify();
}

template <typename TaskType>
uint64 Batch<TaskType>::traceme_context_id() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_schedulerDTh mht_13(mht_13_v, 532, "", "./tensorflow/core/kernels/batching_util/batch_scheduler.h", "Batch<TaskType>::traceme_context_id");

  return traceme_context_id_;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_
