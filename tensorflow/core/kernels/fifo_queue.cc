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
class MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc() {
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

// See docs in ../ops/data_flow_ops.cc.

#include "tensorflow/core/kernels/fifo_queue.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

FIFOQueue::FIFOQueue(int capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : TypedQueue(capacity, component_dtypes, component_shapes, name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::FIFOQueue");
}

void FIFOQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::DequeueLocked");

  DCHECK_GT(queues_[0].size(), size_t{0});
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(queues_[i][0]);
    queues_[i].pop_front();
  }
}

void FIFOQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                           DoneCallback callback) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::TryEnqueue");

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          1, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
              return kComplete;
            }
            if (queues_[0].size() < static_cast<size_t>(capacity_)) {
              for (int i = 0; i < num_components(); ++i) {
                queues_[i].push_back(tuple[i]);
              }
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

/* static */
Status FIFOQueue::GetElementComponentFromBatch(const FIFOQueue::Tuple& tuple,
                                               int64_t index, int component,
                                               OpKernelContext* ctx,
                                               Tensor* out_tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::GetElementComponentFromBatch");

  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  TF_RETURN_IF_ERROR(
      ctx->allocate_temp(tuple[component].dtype(), element_shape, out_tensor));
  TF_RETURN_IF_ERROR(
      batch_util::CopySliceToElement(tuple[component], out_tensor, index));
  return Status::OK();
}

void FIFOQueue::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                               DoneCallback callback) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::TryEnqueueMany");

  const int64_t batch_size = tuple[0].dim_size(0);
  if (batch_size == 0) {
    callback();
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int64_t index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                Tensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return kComplete;
                queues_[i].push_back(element);
              }
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                return kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

void FIFOQueue::TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::TryDequeue");

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_6(mht_6_v, 355, "", "./tensorflow/core/kernels/fifo_queue.cc", "lambda");

            const int64_t queue_size = queues_[0].size();
            if (closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return kComplete;
            }
            if (queue_size > 0) {
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              attempt->done_callback = [callback, tuple]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_7(mht_7_v, 370, "", "./tensorflow/core/kernels/fifo_queue.cc", "lambda");
 callback(tuple); };
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

void FIFOQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                               bool allow_small_batch,
                               CallbackWithTuple callback) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_8(mht_8_v, 391, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::TryDequeueMany");

  if (!specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "FIFOQueue's DequeueMany and DequeueUpTo require the "
        "components to have specified shapes."));
    callback(Tuple());
    return;
  }
  if (num_elements == 0) {
    Tuple tuple;
    tuple.reserve(num_components());
    for (int i = 0; i < num_components(); ++i) {
      // TODO(josh11b,misard): Switch to allocate_output().  Problem is
      // this breaks the abstraction boundary since we don't *really*
      // know if and how the Tensors in the tuple we pass to callback
      // correspond to the outputs of *ctx.  For example, the
      // ReaderRead Op uses TryDequeue() to get a filename out of a
      // queue that is used internally by the reader and is not
      // associated with any output of the ReaderRead.
      // mrry@ adds:
      // Maybe we need to pass a std::function<Tensor*(...)> (or
      // better signature) that calls the appropriate allocator
      // function in addition to ctx?  (Or support a shim Allocator
      // that has an internal OpKernelContext*, and dispatches to the
      // appropriate method?)
      // misard@ adds:
      // I don't see that a std::function would help. The problem is
      // that at this point (allocation time) the system doesn't know
      // what is going to happen to the element read out of the
      // queue. As long as we keep the generality that TensorFlow Ops
      // do their own dynamic allocation in arbitrary C++ code, we
      // need to preserve robustness to allocating output Tensors with
      // the 'wrong' attributes, and fixing up with a copy. The only
      // improvement I can see here in the future would be to support
      // an optimized case where the queue 'knows' what attributes to
      // use, and plumbs them through here.
      Tensor element;
      Status status = ctx->allocate_temp(component_dtypes_[i],
                                         ManyOutShape(i, 0), &element);
      if (!status.ok()) {
        ctx->SetStatus(status);
        callback(Tuple());
        return;
      }
      tuple.emplace_back(element);
    }
    callback(tuple);
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          num_elements, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, allow_small_batch,
           this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_9(mht_9_v, 456, "", "./tensorflow/core/kernels/fifo_queue.cc", "lambda");

            int64_t queue_size = queues_[0].size();

            if (closed_ && queue_size < attempt->elements_requested) {
              // If we don't have enough for a full dequeue, we have
              // to reset the attempt tuple.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the front of the
                // queue.
                for (int64_t i = attempt->tuple[0].dim_size(0) -
                                 attempt->elements_requested - 1;
                     i >= 0; --i) {
                  for (int j = 0; j < num_components(); ++j) {
                    Tensor element;
                    Status s = GetElementComponentFromBatch(
                        attempt->tuple, i, j, attempt->context, &element);
                    if (!s.ok()) {
                      attempt->context->SetStatus(
                          errors::DataLoss("Failed to restore element from "
                                           "partially-dequeued batch "
                                           "to FIFOQueue: ",
                                           s.error_message()));
                    }
                    queues_[j].push_front(element);
                  }
                }
              }
              if (allow_small_batch && !queues_[0].empty()) {
                // Request all remaining elements in the queue.
                queue_size = queues_[0].size();
                attempt->tuple.clear();
                attempt->elements_requested = queue_size;
              } else {
                if (allow_small_batch) {
                  // There may be some other attempts containing
                  // values.  If so, we'll yield and wait for them
                  // to add elements to the queue.
                  if (!enqueue_attempts_.empty()) return kProgress;
                }
                if (attempt->context->status().ok()) {
                  attempt->context->SetStatus(errors::OutOfRange(
                      "FIFOQueue '", name_, "' is closed and has ",
                      "insufficient elements (requested ",
                      attempt->elements_requested, ", current size ",
                      queue_size, ")"));
                }
                return kComplete;
              }
            }

            RunResult result = kNoProgress;
            for (; queue_size > 0; --queue_size) {
              if (attempt->tuple.empty()) {
                // Only allocate tuple when we have something to dequeue
                // so we don't use excessive memory when there are many
                // blocked dequeue attempts waiting.
                attempt->tuple.reserve(num_components());
                for (int i = 0; i < num_components(); ++i) {
                  const TensorShape shape =
                      ManyOutShape(i, attempt->elements_requested);
                  Tensor element;
                  attempt->context->SetStatus(attempt->context->allocate_temp(
                      component_dtypes_[i], shape, &element));
                  if (!attempt->context->status().ok()) return kComplete;
                  attempt->tuple.emplace_back(element);
                }
              }
              result = kProgress;
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              const int64_t index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                attempt->context->SetStatus(batch_util::CopyElementToSlice(
                    std::move(tuple[i]), &attempt->tuple[i], index));
                if (!attempt->context->status().ok()) return kComplete;
              }
              tuple.clear();
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                tuple = attempt->tuple;
                attempt->done_callback = [callback, tuple]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_10(mht_10_v, 540, "", "./tensorflow/core/kernels/fifo_queue.cc", "lambda");

                  callback(tuple);
                };
                return kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

Status FIFOQueue::MatchesNodeDef(const NodeDef& node_def) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_11(mht_11_v, 561, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueue::MatchesNodeDef");

  if (!MatchesNodeDefOp(node_def, "FIFOQueue").ok() &&
      !MatchesNodeDefOp(node_def, "FIFOQueueV2").ok()) {
    return errors::InvalidArgument("Expected FIFOQueue, found ", node_def.op());
  }
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
  return Status::OK();
}

// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
FIFOQueueOp::FIFOQueueOp(OpKernelConstruction* context)
    : TypedQueueOp(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_12(mht_12_v, 580, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueueOp::FIFOQueueOp");

  OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
}

Status FIFOQueueOp::CreateResource(QueueInterface** ret) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfifo_queueDTcc mht_13(mht_13_v, 587, "", "./tensorflow/core/kernels/fifo_queue.cc", "FIFOQueueOp::CreateResource");

  FIFOQueue* queue = new FIFOQueue(capacity_, component_types_,
                                   component_shapes_, cinfo_.name());
  return CreateTypedQueue(queue, ret);
}

}  // namespace tensorflow
