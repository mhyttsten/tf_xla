/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh() {
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


#include <map>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace serving {

// Base class for resource that encapsulating the state and logic for batching
// tensors.
class BatchResourceBase : public ResourceBase {
 public:
  // Given a BatchTask (from one op invocation) with 'num_outputs'== M and
  // splitted into N sub tasks, TensorMatrix is a N X M matrix.
  // Namely, TensorMatrix[i][j] indicates the i-th split tensor of j-th output;
  // concatenating tensors along the 2nd dimension gives a output tensor.
  typedef std::vector<std::vector<Tensor>> TensorMatrix;

  // Ingests data from one invocation of the batch op. The data is enqueued to
  // be combined with others into a batch, asynchronously.
  Status RegisterInput(int64_t guid, OpKernelContext* context,
                       const string& batcher_queue_name,
                       AsyncOpKernel::DoneCallback done_callback);

 public:
  // One task to be batched, corresponds to a `slice` of input from one batch-op
  // invocation.
  //
  // Given input from one batch-op invocation, a `slice` of this input is:
  // 1) Split each Tensor in `BatchTask::inputs` along the 0th dimension.
  // 2) 'split_index' is calculated along the 0-th dimension.
  //
  // Note input from one batch-op invocation is valid and considered a
  // specialized `slice`.
  struct BatchTask : public tensorflow::serving::BatchTask {
    // A unique ID to identify this invocation of Batch.
    int64_t guid;

    Context propagated_context;

    std::vector<Tensor> inputs;
    std::vector<Tensor> captured_inputs;
    OpKernelContext* context;
    AsyncOpKernel::DoneCallback done_callback;

    // The index of this split, along the 0-th dimension of input from op
    // invocation.
    int split_index = 0;

    // Two-dimensional tensor matrix, ownership shared by:
    // 1) each split of task (to fill one row in this matrix)
    // and
    // 2) callback that runs to merge output of individual splits for an op
    // invocation, after all splits complete.
    std::shared_ptr<TensorMatrix> output;

    // 'status' records error (could be from any split) if at least one split
    // returns error, OK otherwise.
    // Ownership is shared by individual splits and callback.
    std::shared_ptr<ThreadSafeStatus> status;

    bool is_partial = false;

    uint64 start_time;

    size_t size() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh mht_0(mht_0_v, 265, "", "./tensorflow/core/kernels/batching_util/batch_resource_base.h", "size");
 return inputs[0].shape().dim_size(0); }

    // Create a split task from this one. The caller needs to setup the inputs
    // of the new task
    std::unique_ptr<BatchTask> CreateSplitTask(
        int split_index, AsyncOpKernel::DoneCallback done_callback);

    // RequestCost is for collecting the cost and must outlive the batching
    // processing.
    //
    // For example, to collect cost in rpc processing, `request_cost` is owned
    // by rpc handler and points to the RequestCost of an rpc which provides
    // the inputs to this BatchTask.
    //
    // After the batch processing, the request cost will be incremented with
    // this task's processing costs.
    RequestCost* request_cost = nullptr;

   protected:
    virtual std::unique_ptr<BatchTask> CreateDerivedTask() {
      return std::make_unique<BatchTask>();
    }
  };

  // Appending a T suffix to make the type alias different to those in
  // tensorflow::serving namespace, because some versions of compiler complain
  // about changing meaning of the symbols.
  using BatcherT = SharedBatchScheduler<BatchResourceBase::BatchTask>;
  using AdaptiveBatcherT =
      AdaptiveSharedBatchScheduler<BatchResourceBase::BatchTask>;
  using BatcherQueueT = BatchScheduler<BatchResourceBase::BatchTask>;
  using BatchT = Batch<BatchResourceBase::BatchTask>;

  BatchResourceBase(bool has_process_batch_function,
                    std::shared_ptr<BatcherT> batcher,
                    const BatcherT::QueueOptions& batcher_queue_options,
                    std::vector<int32> allowed_batch_sizes)
      : has_process_batch_function_(has_process_batch_function),
        batcher_(std::move(batcher)),
        batcher_queue_options_(batcher_queue_options),
        allowed_batch_sizes_(std::move(allowed_batch_sizes)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh mht_1(mht_1_v, 308, "", "./tensorflow/core/kernels/batching_util/batch_resource_base.h", "BatchResourceBase");

    allowed_batch_sizes_str_ = absl::StrJoin(allowed_batch_sizes_, ",");
  }

  BatchResourceBase(bool has_process_batch_function,
                    std::shared_ptr<AdaptiveBatcherT> batcher,
                    const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
                    std::vector<int32> allowed_batch_sizes)
      : has_process_batch_function_(has_process_batch_function),
        adaptive_batcher_(std::move(batcher)),
        adaptive_batcher_queue_options_(batcher_queue_options),
        allowed_batch_sizes_(std::move(allowed_batch_sizes)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSbatch_resource_baseDTh mht_2(mht_2_v, 322, "", "./tensorflow/core/kernels/batching_util/batch_resource_base.h", "BatchResourceBase");
}

  static BatcherT::QueueOptions GetBatcherQueueOptions(
      int32_t num_batch_threads, int32_t max_batch_size,
      int32_t batch_timeout_micros, int32_t max_enqueued_batches,
      const std::vector<int32>& allowed_batch_sizes,
      bool enable_large_batch_splitting);

  static AdaptiveBatcherT::QueueOptions GetAdaptiveBatcherQueueOptions(
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches, bool enable_large_batch_splitting,
      const std::vector<int32>& allowed_batch_sizes);

  // Split 'input' of 'input_task_ptr' along 0th dimension, into a list of
  // 'output_tasks'.
  // Task sizes are determined by
  // 1) open_batch_remaining_slot
  // 2) max_batch_size
  // 3) size-of-input-task
  // in a way that
  // 1) Task sizes add up to `size-of-input-task`.
  // 2) Task sizes from left to right are like
  //    [open_batch_remaining_slot, max_batch_size, max_batch_size, ...,
  //    `size-of-input-task` - `sum-of-previous-elements`].
  //
  // REQUIRES:
  // Caller should make sure size-of-input-task is greater than
  // open_batch_remaining_slot.
  static Status SplitInputTask(
      std::unique_ptr<BatchTask>* input_task_ptr, int open_batch_remaining_slot,
      int max_batch_size,
      std::vector<std::unique_ptr<BatchTask>>* output_tasks);

  // Splits the batch costs to each task.
  //
  // Inputs:
  // 1) batch_cost_measurements, which provides the total cost of each type;
  // 2) processed_size, it's the batch size plus the padding amount;
  // 3) batch, provides the batch size.
  //
  // Outputs:
  // The request_cost in each batch task will be updated. This function will use
  // two approaches to split the batch cost (if it's non-zero), thus two costs
  // will be output.
  // 1) smeared cost: batch cost is split proportionally to each task's size,
  //    and paddings do not share any cost;
  // 2) non-smeared cost: batch cost is split proportionally to each task or
  //    padding's size. Here padding's cost is not assigned to any tasks.
  static void SplitBatchCosts(
      std::vector<std::unique_ptr<CostMeasurement>>& batch_cost_measurements,
      const int64_t processed_size, BatchT& batch);

 private:
  // Implementation of calling the process batch function.
  virtual void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      absl::Span<const Tensor> inputs, std::vector<Tensor>* combined_outputs,
      std::function<void(const Status&)> done) const = 0;

  // Factory method for creating a BatchTask, overridable by subclasses.
  virtual Status CreateBatchTask(
      OpKernelContext* context,
      std::unique_ptr<BatchResourceBase::BatchTask>* output) const;

  // Validates that it's legal to combine the tasks in 'batch' into a batch.
  // Assumes the batch is non-empty.
  static Status ValidateBatch(const BatchT& batch);

  // Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
  // or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
  // returns 'batch_size'.
  int RoundToLowestAllowedBatchSize(int batch_size) const;

  Status ConcatInputTensors(const BatchT& batch, OpKernelContext* context,
                            std::vector<Tensor>* concatenated_tensors) const;

  Status SplitOutputTensors(const std::vector<Tensor>& combined_outputs,
                            BatchT* batch) const;

  void ProcessFuncBatch(std::unique_ptr<BatchT> batch) const;

  // Processes a batch of one or more BatchTask entries.
  void ProcessBatch(std::unique_ptr<BatchT> batch) const;

  // Emits an index tensor, which the Unbatch op will use to un-concatenate
  // the tensor and attribute the pieces to the right batch keys. The index
  // tensor contains, for each input: [batch_key, start_offset, end_offset]
  // where start_offset and end_offset represent the range of entries in the
  // concatenated tensors that belong to that input.
  //
  // Emits the result to the output at 'output_index' using 'context'.
  static Status EmitIndexTensor(OpKernelContext* context, const BatchT& batch,
                                int output_index);

  // Looks up the batcher queue for 'queue_name'. If it did't previously exist,
  // creates it.
  Status LookupOrCreateBatcherQueue(const string& queue_name,
                                    BatcherQueueT** queue);

  // True if user specified a batch processing function for this resource.
  const bool has_process_batch_function_;
  // A batch scheduler, and options for creating queues.
  std::shared_ptr<BatcherT> batcher_;
  BatcherT::QueueOptions batcher_queue_options_;

  // A batch scheduler, and options for creating queues.
  std::shared_ptr<AdaptiveBatcherT> adaptive_batcher_;
  AdaptiveBatcherT::QueueOptions adaptive_batcher_queue_options_;

  // A collection of batcher queues, keyed on queue name.
  // TODO(olston): Garbage-collect unused queues (perhaps simply remove empty
  // ones (with a time delay?); it's okay if they get recreated later).
  mutable mutex batcher_queues_mu_;
  std::map<string, std::unique_ptr<BatcherQueueT>> batcher_queues_
      TF_GUARDED_BY(batcher_queues_mu_);

  std::vector<int32> allowed_batch_sizes_;
  // A concatenated string of <allowed_batch_sizes_>, separated by ",". This is
  // used to record batching parameter.
  string allowed_batch_sizes_str_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_
