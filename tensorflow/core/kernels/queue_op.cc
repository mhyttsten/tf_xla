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
class MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc() {
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

#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

QueueOp::QueueOp(OpKernelConstruction* context) : ResourceOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/queue_op.cc", "QueueOp::QueueOp");

  OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
  if (capacity_ < 0) {
    capacity_ = QueueBase::kUnbounded;
  }
  OP_REQUIRES_OK(context,
                 context->GetAttr("component_types", &component_types_));
}

void QueueOp::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/kernels/queue_op.cc", "QueueOp::Compute");

  ResourceOpKernel<QueueInterface>::Compute(context);
  mutex_lock l(mu_);
  if (resource_ && context->track_allocations()) {
    context->record_persistent_memory_allocation(resource_->MemoryUsed());
  }
}

Status QueueOp::VerifyResource(QueueInterface* queue) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/kernels/queue_op.cc", "QueueOp::VerifyResource");

  return queue->MatchesNodeDef(def());
}


QueueOpKernel::QueueOpKernel(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/kernels/queue_op.cc", "QueueOpKernel::QueueOpKernel");
}

void QueueOpKernel::ComputeAsync(OpKernelContext* ctx, DoneCallback callback) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_4(mht_4_v, 234, "", "./tensorflow/core/kernels/queue_op.cc", "QueueOpKernel::ComputeAsync");

  QueueInterface* queue;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &queue), callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &queue),
                         callback);
  }
  ComputeAsync(ctx, queue, [callback, queue]() {
    queue->Unref();
    callback();
  });
}

QueueAccessOpKernel::QueueAccessOpKernel(OpKernelConstruction* context)
    : QueueOpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/kernels/queue_op.cc", "QueueAccessOpKernel::QueueAccessOpKernel");

  OP_REQUIRES_OK(context, context->GetAttr("timeout_ms", &timeout_));
  // TODO(keveman): Enable timeout.
  OP_REQUIRES(context, timeout_ == -1,
              errors::InvalidArgument("Timeout not supported yet."));
}

// Defines an EnqueueOp, the execution of which enqueues a tuple of
// tensors in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
EnqueueOp::EnqueueOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/kernels/queue_op.cc", "EnqueueOp::EnqueueOp");
}

void EnqueueOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                             DoneCallback callback) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/kernels/queue_op.cc", "EnqueueOp::ComputeAsync");

  DataTypeVector expected_inputs;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    expected_inputs.push_back(DT_RESOURCE);
  } else {
    expected_inputs.push_back(DT_STRING_REF);
  }
  for (DataType dt : queue->component_dtypes()) {
    expected_inputs.push_back(dt);
  }
  OP_REQUIRES_OK_ASYNC(ctx, ctx->MatchSignature(expected_inputs, {}), callback);

  QueueInterface::Tuple tuple;
  OpInputList components;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components),
                       callback);
  for (const Tensor& Tcomponent : components) {
    tuple.push_back(Tcomponent);
  }

  OP_REQUIRES_OK_ASYNC(ctx, queue->ValidateTuple(tuple), callback);
  queue->TryEnqueue(tuple, ctx, callback);
}

// Defines an EnqueueManyOp, the execution of which slices each
// component of a tuple of tensors along the 0th dimension, and
// enqueues tuples of slices in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
//
// N.B. All tuple components must have the same size in the 0th
// dimension.
EnqueueManyOp::EnqueueManyOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_8(mht_8_v, 320, "", "./tensorflow/core/kernels/queue_op.cc", "EnqueueManyOp::EnqueueManyOp");
}

void EnqueueManyOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_9(mht_9_v, 326, "", "./tensorflow/core/kernels/queue_op.cc", "EnqueueManyOp::ComputeAsync");

  DataTypeVector expected_inputs;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    expected_inputs.push_back(DT_RESOURCE);
  } else {
    expected_inputs.push_back(DT_STRING_REF);
  }
  for (DataType dt : queue->component_dtypes()) {
    expected_inputs.push_back(dt);
  }
  OP_REQUIRES_OK_ASYNC(ctx, ctx->MatchSignature(expected_inputs, {}), callback);

  QueueInterface::Tuple tuple;
  OpInputList components;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components),
                       callback);
  for (const Tensor& Tcomponent : components) {
    tuple.push_back(Tcomponent);
  }

  OP_REQUIRES_OK_ASYNC(ctx, queue->ValidateManyTuple(tuple), callback);
  queue->TryEnqueueMany(tuple, ctx, callback);
}

EnqueueManyOp::~EnqueueManyOp() = default;

// Defines a DequeueOp, the execution of which dequeues a tuple of
// tensors from the given Queue.
//
// The op has one input, which is the handle of the appropriate
// Queue. The op has k outputs, where k is the number of components in
// the tuples stored in the given Queue, and output i is the ith
// component of the dequeued tuple.
DequeueOp::DequeueOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_10(mht_10_v, 363, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueOp::DequeueOp");
}

void DequeueOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                             DoneCallback callback) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_11(mht_11_v, 369, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueOp::ComputeAsync");

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature({DT_RESOURCE}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature({DT_STRING_REF}, queue->component_dtypes()),
        callback);
  }

  queue->TryDequeue(ctx, [ctx, callback](const QueueInterface::Tuple& tuple) {
    if (!ctx->status().ok()) {
      callback();
      return;
    }
    OpOutputList output_components;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->output_list("components", &output_components), callback);
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      output_components.set(i, tuple[i]);
    }
    callback();
  });
}

DequeueOp::~DequeueOp() = default;

// Defines a DequeueManyOp, the execution of which concatenates the
// requested number of elements from the given Queue along the 0th
// dimension, and emits the result as a single tuple of tensors.
//
// The op has two inputs:
// - Input 0: the handle to a queue.
// - Input 1: the number of elements to dequeue.
//
// The op has k outputs, where k is the number of components in the
// tuples stored in the given Queue, and output i is the ith component
// of the dequeued tuple.
DequeueManyOp::DequeueManyOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_12(mht_12_v, 412, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueManyOp::DequeueManyOp");
}

void DequeueManyOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_13(mht_13_v, 418, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueManyOp::ComputeAsync");

  const Tensor& Tnum_elements = ctx->input(1);
  int32_t num_elements = Tnum_elements.flat<int32>()(0);

  OP_REQUIRES_ASYNC(ctx, num_elements >= 0,
                    errors::InvalidArgument("DequeueManyOp requested ",
                                            num_elements, " < 0 elements"),
                    callback);

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_RESOURCE, DT_INT32}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->MatchSignature({DT_STRING_REF, DT_INT32},
                                             queue->component_dtypes()),
                         callback);
  }

  queue->TryDequeueMany(
      num_elements, ctx, false /* allow_small_batch */,
      [ctx, callback](const QueueInterface::Tuple& tuple) {
        if (!ctx->status().ok()) {
          callback();
          return;
        }
        OpOutputList output_components;
        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->output_list("components", &output_components), callback);
        for (int i = 0; i < ctx->num_outputs(); ++i) {
          output_components.set(i, tuple[i]);
        }
        callback();
      });
}

DequeueManyOp::~DequeueManyOp() = default;

// Defines a DequeueUpToOp, the execution of which concatenates the
// requested number of elements from the given Queue along the 0th
// dimension, and emits the result as a single tuple of tensors.
//
// The difference between this op and DequeueMany is the handling when
// the Queue is closed.  While the DequeueMany op will return if there
// an error when there are less than num_elements elements left in the
// closed queue, this op will return between 1 and
// min(num_elements, elements_remaining_in_queue), and will not block.
// If there are no elements left, then the standard DequeueMany error
// is returned.
//
// This op only works if the underlying Queue implementation accepts
// the allow_small_batch = true parameter to TryDequeueMany.
// If it does not, an errors::Unimplemented exception is returned.
//
// The op has two inputs:
// - Input 0: the handle to a queue.
// - Input 1: the number of elements to dequeue.
//
// The op has k outputs, where k is the number of components in the
// tuples stored in the given Queue, and output i is the ith component
// of the dequeued tuple.
//
// The op has one attribute: allow_small_batch.  If the Queue supports
// it, setting this to true causes the queue to return smaller
// (possibly zero length) batches when it is closed, up to however
// many elements are available when the op executes.  In this case,
// the Queue does not block when closed.
DequeueUpToOp::DequeueUpToOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_14(mht_14_v, 491, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueUpToOp::DequeueUpToOp");
}

void DequeueUpToOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_15(mht_15_v, 497, "", "./tensorflow/core/kernels/queue_op.cc", "DequeueUpToOp::ComputeAsync");

  const Tensor& Tnum_elements = ctx->input(1);
  int32_t num_elements = Tnum_elements.flat<int32>()(0);

  OP_REQUIRES_ASYNC(ctx, num_elements >= 0,
                    errors::InvalidArgument("DequeueUpToOp requested ",
                                            num_elements, " < 0 elements"),
                    callback);

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_RESOURCE, DT_INT32}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->MatchSignature({DT_STRING_REF, DT_INT32},
                                             queue->component_dtypes()),
                         callback);
  }

  queue->TryDequeueMany(
      num_elements, ctx, true /* allow_small_batch */,
      [ctx, callback](const QueueInterface::Tuple& tuple) {
        if (!ctx->status().ok()) {
          callback();
          return;
        }
        OpOutputList output_components;
        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->output_list("components", &output_components), callback);
        for (int i = 0; i < ctx->num_outputs(); ++i) {
          output_components.set(i, tuple[i]);
        }
        callback();
      });
}

DequeueUpToOp::~DequeueUpToOp() = default;

// Defines a QueueCloseOp, which closes the given Queue. Closing a
// Queue signals that no more elements will be enqueued in it.
//
// The op has one input, which is the handle of the appropriate Queue.
QueueCloseOp::QueueCloseOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_16(mht_16_v, 545, "", "./tensorflow/core/kernels/queue_op.cc", "QueueCloseOp::QueueCloseOp");

  OP_REQUIRES_OK(context, context->GetAttr("cancel_pending_enqueues",
                                           &cancel_pending_enqueues_));
}

void QueueCloseOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                DoneCallback callback) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_17(mht_17_v, 554, "", "./tensorflow/core/kernels/queue_op.cc", "QueueCloseOp::ComputeAsync");

  queue->Close(ctx, cancel_pending_enqueues_, callback);
}

// Defines a QueueSizeOp, which computes the number of elements in the
// given Queue, and emits it as an output tensor.
//
// The op has one input, which is the handle of the appropriate Queue;
// and one output, which is a single-element tensor containing the current
// size of that Queue.
QueueSizeOp::QueueSizeOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_18(mht_18_v, 568, "", "./tensorflow/core/kernels/queue_op.cc", "QueueSizeOp::QueueSizeOp");
}

void QueueSizeOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                               DoneCallback callback) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_19(mht_19_v, 574, "", "./tensorflow/core/kernels/queue_op.cc", "QueueSizeOp::ComputeAsync");

  Tensor* Tqueue_size = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &Tqueue_size));
  Tqueue_size->flat<int32>().setConstant(queue->size());
  callback();
}

QueueIsClosedOp::QueueIsClosedOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_20(mht_20_v, 585, "", "./tensorflow/core/kernels/queue_op.cc", "QueueIsClosedOp::QueueIsClosedOp");
}

void QueueIsClosedOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                   DoneCallback callback) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTcc mht_21(mht_21_v, 591, "", "./tensorflow/core/kernels/queue_op.cc", "QueueIsClosedOp::ComputeAsync");

  Tensor* Tqueue_is_closed = nullptr;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &Tqueue_is_closed));
  Tqueue_is_closed->flat<bool>().setConstant(queue->is_closed());
  callback();
}

}  // namespace tensorflow
