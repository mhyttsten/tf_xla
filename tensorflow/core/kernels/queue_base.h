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

#ifndef TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
#define TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh() {
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
#include <vector>

#include "absl/base/macros.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Functionality common to asynchronous QueueInterface implementations.
class QueueBase : public QueueInterface {
 public:
  // As a possible value of 'capacity'.
  static constexpr int32_t kUnbounded = INT_MAX;

  // Args:
  //   component_dtypes: The types of each component in a queue-element tuple.
  //   component_shapes: The shapes of each component in a queue-element tuple,
  //     which must either be empty (if the shapes are not specified) or
  //     or have the same size as component_dtypes.
  //   name: A name to use for the queue.
  QueueBase(int32_t capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------
  const DataTypeVector& component_dtypes() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/queue_base.h", "component_dtypes");

    return component_dtypes_;
  }

  Status ValidateTuple(const Tuple& tuple) override;
  Status ValidateManyTuple(const Tuple& tuple) override;

  void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             DoneCallback callback) override;

  // Other public methods -----------------------------------------------------
  const std::vector<TensorShape>& component_shapes() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/queue_base.h", "component_shapes");

    return component_shapes_;
  }

  int32 capacity() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/queue_base.h", "capacity");
 return capacity_; }

  bool is_closed() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/kernels/queue_base.h", "is_closed");

    mutex_lock lock(mu_);
    return closed_;
  }

  // Copies the index^th slice (in the first dimension) of parent into element.
  static Status CopySliceToElement(const Tensor& parent, Tensor* element,
                                   int64_t index);

  // Copies element into the index^th slice (in the first dimension) of parent.
  // NOTE(mrry): This method is deprecated. Use
  // `tensorflow::batch_util::CopySliceToElement()` defined in
  // "./batch_util.h" instead.
  ABSL_DEPRECATED(
      "Use `tensorflow::batch_util::CopySliceToElement()` defined in "
      "\"./batch_util.h\" instead.")
  static Status CopyElementToSlice(const Tensor& element, Tensor* parent,
                                   int64_t index);

 protected:
  enum Action { kEnqueue, kDequeue };
  enum RunResult { kNoProgress, kProgress, kComplete };

  // Tries to enqueue/dequeue (or close) based on whatever is at the
  // front of enqueue_attempts_/dequeue_attempts_.  Appends to
  // *finished the callback for any finished attempt (so it may be
  // called once mu_ is released).  Returns true if any progress was
  // made.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_4(mht_4_v, 280, "", "./tensorflow/core/kernels/queue_base.h", "CleanUp");
}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };

  // Returns the number of components in a queue-element tuple.
  int32 num_components() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_5(mht_5_v, 290, "", "./tensorflow/core/kernels/queue_base.h", "num_components");
 return component_dtypes_.size(); }

  // True if shapes were specified.  If so, inputs will be validated
  // against them, etc.
  bool specified_shapes() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/queue_base.h", "specified_shapes");
 return component_shapes_.size() > 0; }

  // Code common to Validate*Tuple().
  Status ValidateTupleCommon(const Tuple& tuple) const;

  TensorShape ManyOutShape(int i, int64_t batch_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/queue_base.h", "ManyOutShape");

    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

  void Cancel(Action action, CancellationManager* cancellation_manager,
              CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  ~QueueBase() override;

  // Helpers for implementing MatchesNodeDef().
  static string ShapeListString(const gtl::ArraySlice<TensorShape>& shapes);
  Status MatchesNodeDefOp(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefCapacity(const NodeDef& node_def,
                                int32_t capacity) const;
  Status MatchesNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesNodeDefShapes(const NodeDef& node_def) const;

 protected:
  const int32 capacity_;
  const DataTypeVector component_dtypes_;
  const std::vector<TensorShape> component_shapes_;
  const string name_;
  mutable mutex mu_;
  bool closed_ TF_GUARDED_BY(mu_);

  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int32 elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationManager* cancellation_manager;  // not owned
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;
    Tuple tuple;
    // tuples is used by some implementations allowing dynamic shapes.
    std::vector<Tuple> tuples;

    Attempt(int32_t elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationManager* cancellation_manager,
            CancellationToken cancellation_token, RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(done_callback),
          context(context),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token),
          run_callback(run_callback),
          is_cancelled(false) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTh mht_8(mht_8_v, 369, "", "./tensorflow/core/kernels/queue_base.h", "Attempt");
}
  };
  std::deque<Attempt> enqueue_attempts_ TF_GUARDED_BY(mu_);
  std::deque<Attempt> dequeue_attempts_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(QueueBase);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
