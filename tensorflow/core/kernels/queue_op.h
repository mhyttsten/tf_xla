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

#ifndef TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_
#define TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTh() {
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


#include <deque>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Defines a QueueOp, an abstract class for Queue construction ops.
class QueueOp : public ResourceOpKernel<QueueInterface> {
 public:
  QueueOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 protected:
  // Variables accessible by subclasses
  int32 capacity_;
  DataTypeVector component_types_;

 private:
  Status VerifyResource(QueueInterface* queue) override;
};

class TypedQueueOp : public QueueOp {
 public:
  using QueueOp::QueueOp;

 protected:
  template <typename TypedQueue>
  Status CreateTypedQueue(TypedQueue* queue, QueueInterface** ret) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_opDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/queue_op.h", "CreateTypedQueue");

    if (queue == nullptr) {
      return errors::ResourceExhausted("Failed to allocate queue.");
    }
    *ret = queue;
    return queue->Initialize();
  }
};

// Queue manipulator kernels

class QueueOpKernel : public AsyncOpKernel {
 public:
  explicit QueueOpKernel(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final;

 protected:
  virtual void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                            DoneCallback callback) = 0;
};

class QueueAccessOpKernel : public QueueOpKernel {
 public:
  explicit QueueAccessOpKernel(OpKernelConstruction* context);

 protected:
  int64_t timeout_;
};

// Defines an EnqueueOp, the execution of which enqueues a tuple of
// tensors in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
class EnqueueOp : public QueueAccessOpKernel {
 public:
  explicit EnqueueOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EnqueueOp);
};

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
class EnqueueManyOp : public QueueAccessOpKernel {
 public:
  explicit EnqueueManyOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~EnqueueManyOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EnqueueManyOp);
};

// Defines a DequeueOp, the execution of which dequeues a tuple of
// tensors from the given Queue.
//
// The op has one input, which is the handle of the appropriate
// Queue. The op has k outputs, where k is the number of components in
// the tuples stored in the given Queue, and output i is the ith
// component of the dequeued tuple.
class DequeueOp : public QueueAccessOpKernel {
 public:
  explicit DequeueOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DequeueOp);
};

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
class DequeueManyOp : public QueueAccessOpKernel {
 public:
  explicit DequeueManyOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueManyOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DequeueManyOp);
};

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
class DequeueUpToOp : public QueueAccessOpKernel {
 public:
  explicit DequeueUpToOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueUpToOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DequeueUpToOp);
};

// Defines a QueueCloseOp, which closes the given Queue. Closing a
// Queue signals that no more elements will be enqueued in it.
//
// The op has one input, which is the handle of the appropriate Queue.
class QueueCloseOp : public QueueOpKernel {
 public:
  explicit QueueCloseOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  bool cancel_pending_enqueues_;
  TF_DISALLOW_COPY_AND_ASSIGN(QueueCloseOp);
};

// Defines a QueueSizeOp, which computes the number of elements in the
// given Queue, and emits it as an output tensor.
//
// The op has one input, which is the handle of the appropriate Queue;
// and one output, which is a single-element tensor containing the current
// size of that Queue.
class QueueSizeOp : public QueueOpKernel {
 public:
  explicit QueueSizeOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(QueueSizeOp);
};

class QueueIsClosedOp : public QueueOpKernel {
 public:
  explicit QueueIsClosedOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(QueueIsClosedOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_
