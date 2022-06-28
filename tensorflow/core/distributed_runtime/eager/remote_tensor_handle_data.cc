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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"

#include "tensorflow/core/distributed_runtime/eager/destroy_tensor_handle_node.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

namespace {

void DestroyRemoteTensorHandle(EagerContext* ctx, const string& remote_task,
                               uint64 context_id, uint64 op_id, int output_num,
                               bool ready) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("remote_task: \"" + remote_task + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "DestroyRemoteTensorHandle");

  if (ctx->GetContextId() != context_id) {
    // This means that this tensor was pointing to a remote device, which
    // has been changed out from under us. Simply return since there is
    // nothing we can do.
    return;
  }

  core::RefCountPtr<eager::EagerClient> eager_client;
  Status status = ctx->GetClient(remote_task, &eager_client);
  if (!status.ok()) {
    LOG_EVERY_N_SEC(INFO, 60)
        << "Unable to destroy remote tensor handle because the target "
        << remote_task << " is no longer available.";
    return;
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  auto* handle_to_decref = request->add_queue()->mutable_handle_to_decref();
  handle_to_decref->set_op_id(op_id);
  handle_to_decref->set_output_num(output_num);

  VLOG(3) << "Sending request to delete " << request->DebugString();
  std::unique_ptr<EagerNode> node(
      absl::make_unique<eager::DestroyTensorHandleNode>(
          std::move(request), std::move(eager_client), ready));
  auto& executor = ctx->Executor();
  if (executor.Async()) {
    Status status = executor.AddOrExecute(std::move(node));
    if (!status.ok()) {
      LOG_EVERY_N_SEC(WARNING, 60)
          << "Unable to destroy remote tensor handles. If you are "
             "running a tf.function, it usually indicates some op in "
             "the graph gets an error: "
          << status.error_message();
    }
  } else {
    // This thread may still hold tensorflow::StreamingRPCState::mu_. We need
    // to send out the destroy request in a new thread to avoid deadlock.
    auto* released_node = node.release();
    (*ctx->runner())([ctx, released_node] {
      Status status =
          ctx->Executor().AddOrExecute(absl::WrapUnique(released_node));
      if (!status.ok()) {
        LOG_EVERY_N_SEC(WARNING, 60)
            << "Unable to destroy remote tensor handles. If you are "
               "running a tf.function, it usually indicates some op in "
               "the graph gets an error: "
            << status.error_message();
      }
    });
  }
}
}  // namespace

RemoteTensorHandleData::RemoteTensorHandleData(int64_t op_id, int output_num,
                                               uint64 context_view_id,
                                               bool is_ready)
    : is_ready_(is_ready),
      op_id_(op_id),
      output_num_(output_num),
      context_view_id_(context_view_id),
      ctx_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_1(mht_1_v, 267, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::RemoteTensorHandleData");

  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
}

RemoteTensorHandleData::RemoteTensorHandleData(int64_t op_id, int output_num,
                                               const string& remote_task,
                                               EagerContext* ctx)
    : is_ready_(false),
      op_id_(op_id),
      output_num_(output_num),
      remote_task_(remote_task),
      context_id_(ctx->GetContextId()),
      context_view_id_(ctx->GetContextViewId()),
      ctx_(ctx) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("remote_task: \"" + remote_task + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::RemoteTensorHandleData");

  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
  ctx_->Ref();
}

RemoteTensorHandleData::~RemoteTensorHandleData() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_3(mht_3_v, 296, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::~RemoteTensorHandleData");

  if (ctx_) {
    DestroyRemoteTensorHandle(ctx_, remote_task_, context_id_, op_id_,
                              output_num_, /*ready=*/true);
    ctx_->Unref();
  }
}

Status RemoteTensorHandleData::Shape(TensorShape* shape) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_4(mht_4_v, 307, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::Shape");

  TF_RETURN_IF_ERROR(WaitReady("Shape"));

  tf_shared_lock l(mu_);
  *shape = shape_;

  return Status::OK();
}

Status RemoteTensorHandleData::NumDims(int* num_dims) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::NumDims");

  TF_RETURN_IF_ERROR(WaitReady("NumDims"));

  tf_shared_lock l(mu_);
  *num_dims = shape_.dims();

  return Status::OK();
}

Status RemoteTensorHandleData::Dim(int dim_index, int64_t* dim) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_6(mht_6_v, 331, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::Dim");

  TF_RETURN_IF_ERROR(WaitReady("Dim"));

  tf_shared_lock l(mu_);
  *dim = shape_.dim_size(dim_index);

  return Status::OK();
}

Status RemoteTensorHandleData::NumElements(int64_t* num_elements) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_7(mht_7_v, 343, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::NumElements");

  TF_RETURN_IF_ERROR(WaitReady("NumElements"));

  tf_shared_lock l(mu_);
  *num_elements = shape_.num_elements();

  return Status::OK();
}

bool RemoteTensorHandleData::IsReady() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_8(mht_8_v, 355, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::IsReady");

  tf_shared_lock l(mu_);
  return is_ready_;
}

void RemoteTensorHandleData::Poison(Status status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_9(mht_9_v, 363, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::Poison");

  mutex_lock l(mu_);
  is_poisoned_ = status;
  is_ready_ = true;
}

Status RemoteTensorHandleData::IsPoisoned() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_10(mht_10_v, 372, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::IsPoisoned");

  tf_shared_lock l(mu_);
  return is_poisoned_;
}

Status RemoteTensorHandleData::SetShape(const TensorShape& shape) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_11(mht_11_v, 380, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::SetShape");

  return SetShapeAndRemoteTask(shape, /*remote_task=*/"");
}

Status RemoteTensorHandleData::SetShapeAndRemoteTask(
    const TensorShape& shape, const string& remote_task) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("remote_task: \"" + remote_task + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_12(mht_12_v, 389, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::SetShapeAndRemoteTask");

  // If `is_ready_` is set previously due to poisoning, return the original
  // error that poisoned this tensor.
  TF_RETURN_IF_ERROR(IsPoisoned());

  mutex_lock l(mu_);
  if (is_ready_) {
    return errors::Internal("SetShape is only called on non-ready handles.");
  }

  shape_ = shape;
  if (!remote_task.empty()) {
    remote_task_ = remote_task;
  }
  is_poisoned_ = Status::OK();
  is_ready_ = true;

  return Status::OK();
}

string RemoteTensorHandleData::DebugString() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_13(mht_13_v, 412, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::DebugString");

  return strings::StrCat("RemoteTensorHandleData:", " op_id: ", op_id_,
                         " output_num: ", output_num_);
}

Status RemoteTensorHandleData::OpIdAndOutputNum(const bool wait_util_ready,
                                                int64_t* op_id,
                                                int32* output_num) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_14(mht_14_v, 422, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::OpIdAndOutputNum");

  if (wait_util_ready) {
    TF_RETURN_IF_ERROR(WaitReady("OpIdAndOutputNumUntilReady"));
  }
  *op_id = op_id_;
  *output_num = output_num_;
  return Status::OK();
}

Status RemoteTensorHandleData::WaitReady(const char* caller) const {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_tensor_handle_dataDTcc mht_15(mht_15_v, 435, "", "./tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.cc", "RemoteTensorHandleData::WaitReady");

  tf_shared_lock l(mu_);
  if (!is_ready_) {
    profiler::TraceMe activity(
        [caller] { return absl::StrCat(caller, " WaitReady"); },
        profiler::TraceMeLevel::kInfo);
    DVLOG(3) << "WaitReady: " << caller << " " << this;
    mu_.Await(Condition(&is_ready_));
  }
  return is_poisoned_;
}

}  // namespace tensorflow
