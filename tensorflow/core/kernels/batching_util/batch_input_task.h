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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh() {
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


#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/container/fixed_array.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/incremental_barrier.h"

namespace tensorflow {
namespace serving {

namespace internal {
template <typename TaskType>
class BatchInputTaskHandleTestAccess;

template <typename TaskType>
class BatchInputTaskTestAccess;

template <typename TaskType>
class BatchInputTask;

// A RAII-style object that holds a ref-counted batch-input-task, and
// represents a slice of batch-input-task.

// To be handed out to callers of `BatchInputTask::ToTaskHandles` quickly
// (i.e. not necessarily waiting for input split)
//
// `BatchInputTaskHandle::GetSplitTask` evaluates to the slice of task.
template <typename TaskType>
class BatchInputTaskHandle : public BatchTask {
 public:
  BatchInputTaskHandle(
      std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id,
      size_t task_size);

  // Should be called once. Returns nullptr on subsequent calls.
  std::unique_ptr<TaskType> GetSplitTask();

  // Returns the size of this task.
  size_t size() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "size");
 return task_size_; }

 private:
  template <typename T>
  friend class internal::BatchInputTaskHandleTestAccess;

  int split_id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_1(mht_1_v, 244, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "split_id");
 return split_id_; }

  std::shared_ptr<BatchInputTask<TaskType>> batch_input_task_;

  // The handle evaluates to the N-th slice of original task, and
  // N is `split_id_`.
  const int split_id_;

  const size_t task_size_;

  std::atomic<bool> once_{false};
};

// BatchInputTask encapsulates a input (`input_task`) to be batched and the
// information to get task splits after it's enqueued, so as to support lazy
// split of a task.
//
// Input split could reduce excessive padding for efficiency; lazy split
// moves task-split out of the critical path of enqueue and dequeue and reduces
// contention.
//
// BatchInputTask is thread safe.
//
// Usage
//
// ... a deque with frequent enqueue and dequeue operations ...
// ... Note, a deque of Batch of BatchInputTaskHandle is used to form batches
//     at enqueue time (split is lazy at deque time);
// ... For use cases to form batches at dequeue time, we can use a deque of
//     BatchInputTaskHandle directly, and "peek" metadata to form a batch by
//     then.
// std::deque<std::unique_ptr<Batch<BatchInputTaskHandle<TaskType>>>> deque_
//     TF_GUARDED_BY(mu_);
//
// std::unique_ptr<TaskType> input_task;
//
// ... Enqueue path ...
//
// {
//   mutex_lock l(mu_);
//   std::shared_ptr<BatchInputTask<TaskType>> batch_input_task =
//       ConstructLazyBatchWithoutSplit(input_task);
//
//   std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>> task_handles;
//   input_batch->ToTaskHandles(&task_handles);
//   for (int i = 0; i < task_handles.size(); ++i) {
//     EnqueueTaskHandleIntoDeque(deque_);
//   }
//
// ... Dequeue path ...
// std::unique_ptr<Batch<BatchInputTaskHandle<TaskType>>> handles_to_schedule;
// {
//    mutex_lock l(mu_);
//    ... HasBatchToSchedule could be customized or specialized
//    ... (e.g., readiness depending on enqueue time)
//    if (HasBatchToSchedule(deque_)) {
//      handles_to_schedule = std::move(deque_.front());
//      deque_.pop_front();
//    }
// }
// ...... `mu_` is released ......
//
// std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>> tasks_in_batch =
//     RemoveAllTasksFromBatch(handles_to_schedule);
//
// std::unique_ptr<Batch<TaskType>> batch_to_schedule;
// for (int i = 0; i < tasks_in_batch.size(); i++) {
//   batch_to_schedule->AddTask(std::move(tasks_in_batch[i]->GetSplitTask()));
// }
// batch_to_schedule->Close();
//
// `batch_to_schedule` is ready for schedule.
template <typename TaskType>
class BatchInputTask
    : public std::enable_shared_from_this<BatchInputTask<TaskType>> {
 public:
  using SplitInputFunc = std::function<Status(
      std::unique_ptr<TaskType>* input_task, int first_output_task_size,
      int input_batch_size_limit,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;

  BatchInputTask(std::unique_ptr<TaskType> input_task,
                 int open_batch_remaining_slot, int batch_size_limit,
                 SplitInputFunc split_input_func);

  // Outputs the task handles for the input task.
  // Each task handle represents a slice of task after input task is split, and
  // could evaluate to that slice.
  //
  // NOTE:
  // Each task handle in `output_task_handles` takes ownership of a reference of
  // this BatchInputTask.
  void ToTaskHandles(
      std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>>*
          output_task_handles);

 private:
  friend class BatchInputTaskHandle<TaskType>;
  template <typename T>
  friend class internal::BatchInputTaskTestAccess;

  std::unique_ptr<TaskType> GetSplitTask(int split_id);

  Status SplitBatches(std::vector<std::unique_ptr<TaskType>>* output_tasks);

  std::unique_ptr<TaskType> input_task_;

  const int input_task_size_ = 0;
  const int open_batch_remaining_slot_;

  const int batch_size_limit_;
  const SplitInputFunc split_func_;

  const InputSplitMetadata input_split_metadata_;

  mutable absl::once_flag once_;

  std::vector<std::unique_ptr<TaskType>> task_splits_;
  Status split_status_;
};

//
// Implementation details. API readers may skip.
//

template <typename TaskType>
BatchInputTaskHandle<TaskType>::BatchInputTaskHandle(
    std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id,
    size_t task_size)
    : batch_input_task_(batch_input_task),
      split_id_(split_id),
      task_size_(task_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_2(mht_2_v, 378, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTaskHandle<TaskType>::BatchInputTaskHandle");
}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTaskHandle<TaskType>::GetSplitTask() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_3(mht_3_v, 384, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTaskHandle<TaskType>::GetSplitTask");

  if (once_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  once_.store(true, std::memory_order_release);
  return batch_input_task_->GetSplitTask(split_id_);
}

template <typename TaskType>
BatchInputTask<TaskType>::BatchInputTask(std::unique_ptr<TaskType> input_task,
                                         int open_batch_remaining_slot,
                                         int batch_size_limit,
                                         SplitInputFunc split_input_func)
    : input_task_(std::move(input_task)),
      input_task_size_(input_task_->size()),
      open_batch_remaining_slot_(open_batch_remaining_slot),
      batch_size_limit_(batch_size_limit),
      split_func_(split_input_func),
      input_split_metadata_(input_task_size_, open_batch_remaining_slot,
                            batch_size_limit) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_4(mht_4_v, 406, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTask<TaskType>::BatchInputTask");
}

template <typename TaskType>
void BatchInputTask<TaskType>::ToTaskHandles(
    std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>>*
        task_handles) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_5(mht_5_v, 414, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTask<TaskType>::ToTaskHandles");

  const absl::FixedArray<int>& task_sizes = input_split_metadata_.task_sizes();
  task_handles->resize(task_sizes.size());
  for (int i = 0; i < task_handles->size(); i++) {
    (*task_handles)[i] = std::make_unique<BatchInputTaskHandle<TaskType>>(
        this->shared_from_this(), i, task_sizes[i]);
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTask<TaskType>::GetSplitTask(int split_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_6(mht_6_v, 427, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTask<TaskType>::GetSplitTask");

  absl::call_once(once_,
                  [this]() { split_status_ = SplitBatches(&task_splits_); });
  if (!split_status_.ok()) {
    LOG_EVERY_N_SEC(WARNING, 60 /* seconds */)
        << "Split task with error: " << split_status_ << " split metadata is "
        << input_split_metadata_.DebugString();
    return nullptr;
  }
  if (split_id >= 0 && split_id < task_splits_.size()) {
    return std::move(task_splits_[split_id]);
  }
  return nullptr;
}

template <typename TaskType>
Status BatchInputTask<TaskType>::SplitBatches(
    std::vector<std::unique_ptr<TaskType>>* output_tasks) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_input_taskDTh mht_7(mht_7_v, 447, "", "./tensorflow/core/kernels/batching_util/batch_input_task.h", "BatchInputTask<TaskType>::SplitBatches");

  return split_func_(&input_task_, open_batch_remaining_slot_,
                     batch_size_limit_, output_tasks);
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
