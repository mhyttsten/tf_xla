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
class MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc() {
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

#include "tensorflow/core/kernels/queue_base.h"

#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

namespace {

template <DataType DT>
Status HandleSliceToElement(const Tensor& parent, Tensor* element,
                            int64_t index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/queue_base.cc", "HandleSliceToElement");

  typedef typename EnumToDataType<DT>::Type T;
  DCHECK_NE(parent.dim_size(0), 0);
  DCHECK_GE(index, 0);
  if (element->NumElements() != (parent.NumElements() / parent.dim_size(0))) {
    TensorShape chip_shape = parent.shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleSliceToElement Cannot copy slice: number of elements does not "
        "match.  Shapes are: [element]: ",
        element->shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  auto parent_as_matrix = parent.flat_outer_dims<T>();
  element->flat<T>() = parent_as_matrix.chip(index, 0);
  return Status::OK();
}

}  // namespace

QueueBase::QueueBase(int32_t capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : capacity_(capacity),
      component_dtypes_(component_dtypes),
      component_shapes_(component_shapes),
      name_(name),
      closed_(false) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::QueueBase");
}

QueueBase::~QueueBase() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::~QueueBase");
}

Status QueueBase::ValidateTupleCommon(const Tuple& tuple) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_3(mht_3_v, 242, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::ValidateTupleCommon");

  if (tuple.size() != static_cast<size_t>(num_components())) {
    return errors::InvalidArgument(
        "Wrong number of components in tuple. Expected ", num_components(),
        ", got ", tuple.size());
  }
  for (size_t i = 0; i < tuple.size(); ++i) {
    if (tuple[i].dtype() != component_dtypes_[i]) {
      return errors::InvalidArgument(
          "Type mismatch in tuple component ", i, ". Expected ",
          DataTypeString(component_dtypes_[i]), ", got ",
          DataTypeString(tuple[i].dtype()));
    }
  }
  return Status::OK();
}

// static
string QueueBase::ShapeListString(const gtl::ArraySlice<TensorShape>& shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_4(mht_4_v, 263, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::ShapeListString");

  string result = "[";
  bool first = true;
  for (const TensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

Status QueueBase::MatchesNodeDefOp(const NodeDef& node_def,
                                   const string& op) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_5(mht_5_v, 279, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::MatchesNodeDefOp");

  if (node_def.op() != op) {
    return errors::InvalidArgument("Shared queue '", name_, "' has type '", op,
                                   "' that does not match type of Node '",
                                   node_def.name(), "': ", node_def.op());
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefCapacity(const NodeDef& node_def,
                                         int32_t capacity) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_6(mht_6_v, 292, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::MatchesNodeDefCapacity");

  int32_t requested_capacity = -1;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "capacity", &requested_capacity));
  if (requested_capacity < 0) requested_capacity = kUnbounded;
  if (requested_capacity != capacity) {
    return errors::InvalidArgument("Shared queue '", name_, "' has capacity ",
                                   capacity, " but requested capacity was ",
                                   requested_capacity);
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefTypes(const NodeDef& node_def) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::MatchesNodeDefTypes");

  DataTypeVector requested_dtypes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "component_types", &requested_dtypes));
  if (requested_dtypes != component_dtypes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component types ",
                                   DataTypeSliceString(component_dtypes_),
                                   " but requested component types were ",
                                   DataTypeSliceString(requested_dtypes));
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefShapes(const NodeDef& node_def) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_8(mht_8_v, 324, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::MatchesNodeDefShapes");

  std::vector<TensorShape> requested_shapes;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "shapes", &requested_shapes));
  if (requested_shapes != component_shapes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component shapes ",
                                   ShapeListString(component_shapes_),
                                   " but requested component shapes were ",
                                   ShapeListString(requested_shapes));
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status QueueBase::ValidateTuple(const Tuple& tuple) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_9(mht_9_v, 342, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::ValidateTuple");

  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (!component_shapes_[i].IsSameSize(tuple[i].shape())) {
        return errors::InvalidArgument(
            "Shape mismatch in tuple component ", i, ". Expected ",
            component_shapes_[i].DebugString(), ", got ",
            tuple[i].shape().DebugString());
      }
    }
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status QueueBase::ValidateManyTuple(const Tuple& tuple) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_10(mht_10_v, 362, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::ValidateManyTuple");

  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  const int64_t batch_size = tuple[0].dim_size(0);
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      // Expected shape is [batch_size] + component_shapes_[i]
      const TensorShape expected_shape = ManyOutShape(i, batch_size);
      if (!expected_shape.IsSameSize(tuple[i].shape())) {
        return errors::InvalidArgument("Shape mismatch in tuple component ", i,
                                       ". Expected ",
                                       expected_shape.DebugString(), ", got ",
                                       tuple[i].shape().DebugString());
      }
    }
  } else {
    for (size_t i = 1; i < tuple.size(); ++i) {
      if (tuple[i].dim_size(0) != batch_size) {
        return errors::InvalidArgument(
            "All input tensors must have the same size in the 0th ",
            "dimension. Component ", i, " has ", tuple[i].dim_size(0),
            ", and should have ", batch_size);
      }
    }
  }
  return Status::OK();
}

void QueueBase::Cancel(Action action, CancellationManager* cancellation_manager,
                       CancellationToken token) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_11(mht_11_v, 393, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::Cancel");

  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);
    std::deque<Attempt>* attempts =
        action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

    for (Attempt& attempt : *attempts) {
      if (attempt.cancellation_manager == cancellation_manager &&
          attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          if (action == kEnqueue) {
            attempt.context->SetStatus(
                errors::Cancelled("Enqueue operation was cancelled"));
          } else {
            attempt.context->SetStatus(
                errors::Cancelled("Dequeue operation was cancelled"));
          }
          std::swap(callback, attempt.done_callback);
        }
        break;
      }
    }
  }
  if (callback) {
    callback();
    FlushUnlocked();
  }
}

void QueueBase::CloseAndCancel() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_12(mht_12_v, 427, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::CloseAndCancel");

  std::vector<DoneCallback> callbacks;
  {
    mutex_lock lock(mu_);
    closed_ = true;
    for (Attempt& attempt : enqueue_attempts_) {
      if (!attempt.is_cancelled) {
        attempt.is_cancelled = true;
        attempt.context->SetStatus(
            errors::Cancelled("Enqueue operation was cancelled"));
        callbacks.emplace_back(std::move(attempt.done_callback));
      }
    }
  }
  for (const DoneCallback& callback : callbacks) {
    callback();
  }
  FlushUnlocked();
}

void QueueBase::Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                      DoneCallback callback) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_13(mht_13_v, 451, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::Close");

  if (cancel_pending_enqueues) {
    CloseAndCancel();
    callback();
  } else {
    {
      mutex_lock lock(mu_);
      enqueue_attempts_.emplace_back(
          0, callback, ctx, nullptr, CancellationManager::kInvalidToken,
          [this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("Queue '", name_, "' is already closed."));
            } else {
              closed_ = true;
            }
            return kComplete;
          });
    }
    FlushUnlocked();
  }
}

bool QueueBase::TryAttemptLocked(Action action,
                                 std::vector<CleanUp>* clean_up) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_14(mht_14_v, 478, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::TryAttemptLocked");

  std::deque<Attempt>* attempts =
      action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

  bool progress = false;
  bool done = false;
  while (!done && !attempts->empty()) {
    if (attempts->front().is_cancelled) {
      if (action == kEnqueue) {
        if (closed_) {
          VLOG(1) << "Skipping cancelled enqueue attempt";
        } else {
          LOG(WARNING)
              << name_
              << ": Skipping cancelled enqueue attempt with queue not closed";
        }
      } else {
        if (closed_) {
          VLOG(1) << "Skipping cancelled dequeue attempt";
        } else {
          LOG(WARNING)
              << name_
              << ": Skipping cancelled dequeue attempt with queue not closed";
        }
      }
      attempts->pop_front();
    } else {
      Attempt* cur_attempt = &attempts->front();
      switch (cur_attempt->run_callback(cur_attempt)) {
        case kNoProgress:
          done = true;
          break;
        case kProgress:
          done = true;
          progress = true;
          break;
        case kComplete:
          progress = true;
          clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                 cur_attempt->cancellation_token,
                                 cur_attempt->context->cancellation_manager());
          attempts->pop_front();
          break;
      }
    }
  }
  return progress;
}

void QueueBase::FlushUnlocked() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_15(mht_15_v, 530, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::FlushUnlocked");

  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(kEnqueue, &clean_up);
      changed = TryAttemptLocked(kDequeue, &clean_up) || changed;
    } while (changed);
  }
  Unref();
  for (const auto& to_clean : clean_up) {
    if (to_clean.to_deregister != CancellationManager::kInvalidToken) {
      // NOTE(mrry): We can safely ignore the return value of
      // DeregisterCallback because the mutex mu_ ensures that the
      // cleanup action only executes once.
      to_clean.cm->DeregisterCallback(to_clean.to_deregister);
    }
    to_clean.finished();
  }
}

Status QueueBase::CopySliceToElement(const Tensor& parent, Tensor* element,
                                     int64_t index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_16(mht_16_v, 557, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::CopySliceToElement");

  return batch_util::CopySliceToElement(parent, element, index);
}

/* static */
Status QueueBase::CopyElementToSlice(const Tensor& element, Tensor* parent,
                                     int64_t index) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSqueue_baseDTcc mht_17(mht_17_v, 566, "", "./tensorflow/core/kernels/queue_base.cc", "QueueBase::CopyElementToSlice");

  return batch_util::CopyElementToSlice(element, parent, index);
}

}  // namespace tensorflow
