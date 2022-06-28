/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh() {
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


#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// A TensorFlow Eager Worker runs ops and supports worker to worker
// Tensor transfer.
//
// See eager_service.proto for more details about each method.
// This class can be wrapped by specific classes that implement rpc transports
// over this (e.g. gRPC).
class EagerServiceImpl {
 public:
  explicit EagerServiceImpl(const WorkerEnv* env) : env_(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "EagerServiceImpl");

    gc_thread_.reset(
        env_->env->StartThread({}, "EagerServiceContextGC", [this]() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "lambda");

          while (true) {
            {
              mutex_lock l(gc_thread_shutdown_mu_);
              gc_thread_cv_.wait_for(l, std::chrono::seconds(1));

              if (shutting_down_) {
                return;
              }
            }
            {
              mutex_lock l(contexts_mu_);
              for (auto it = contexts_.begin(); it != contexts_.end();) {
                if (it->second->IsStale()) {
                  it->second->Unref();
                  it = contexts_.erase(it);
                } else {
                  it++;
                }
              }
            }
          }
        }));
  }
  virtual ~EagerServiceImpl() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "~EagerServiceImpl");

    {
      mutex_lock l(gc_thread_shutdown_mu_);
      shutting_down_ = true;
      gc_thread_cv_.notify_all();
    }
    gc_thread_.reset();

    mutex_lock l(contexts_mu_);
    for (auto& entry : contexts_) {
      entry.second->Unref();
    }
  }

  Status CreateContext(const CreateContextRequest* request,
                       CreateContextResponse* response);

  Status UpdateContext(const UpdateContextRequest* request,
                       UpdateContextResponse* response);

  // Create a ServerContext for master eager context.
  Status CreateMasterContext(const tensorflow::uint64 context_id,
                             EagerContext* context);

  static constexpr uint64 kInvalidStreamId = 0;

  // Used by both Enqueue and StreamingEnqueue RPCs.
  Status Enqueue(CallOptions* call_opts, const EnqueueRequest* request,
                 EnqueueResponse* response,
                 uint64 stream_id = kInvalidStreamId);

  Status WaitQueueDone(const WaitQueueDoneRequest* request,
                       WaitQueueDoneResponse* response);

  void RunComponentFunction(CallOptions* call_opts,
                            const RunComponentFunctionRequest* request,
                            RunComponentFunctionResponse* response,
                            StatusCallback done);

  Status KeepAlive(const KeepAliveRequest* request,
                   KeepAliveResponse* response);

  Status CloseContext(const CloseContextRequest* request,
                      CloseContextResponse* response);

 protected:
  // This is the server-side execution context. All state regarding execution of
  // a client's ops is held in this server-side context (all generated tensors,
  // and the EagerContext).
  class ServerContext : public core::RefCounted {
   public:
    // Create a ServerContext for local master.
    static ServerContext* CreateMasterContext(tensorflow::EagerContext* ctx,
                                              const WorkerEnv* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_3(mht_3_v, 297, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "CreateMasterContext");

      return new ServerContext(ctx, -1, env, /* is_master= */ true);
    }

    explicit ServerContext(tensorflow::EagerContext* ctx,
                           int64_t destroy_after_secs, const WorkerEnv* env,
                           const bool is_master = false)
        : ctx_(ctx), env_(env), is_master_(is_master) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_4(mht_4_v, 307, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "ServerContext");

      ctx->Ref();
      destroy_after_micros_ =
          destroy_after_secs * tensorflow::EnvTime::kSecondsToMicros;
      RecordAccess();
    }

    ~ServerContext() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_5(mht_5_v, 317, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "~ServerContext");

      // TFE_Context is responsible for shutting down master eager context.
      if (!is_master_) {
        ctx_->WaitForAndCloseRemoteContexts();
      }
      // ctx_->RefCountIsOne() should be true here when is_master_ = false.
      // TODO(iga): Remove EagerContext refcounting.
      ctx_->Unref();
    }

    tensorflow::EagerContext* Context() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_6(mht_6_v, 330, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "Context");
 return ctx_; }

    void RecordAccess() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_7(mht_7_v, 335, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "RecordAccess");

      mutex_lock l(last_accessed_mu_);
      last_accessed_micros_ = env_->env->NowMicros();
    }

    bool IsStale() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_8(mht_8_v, 343, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "IsStale");

      mutex_lock l(last_accessed_mu_);
      const int64_t time_passed =
          env_->env->NowMicros() - last_accessed_micros_;
      return (destroy_after_micros_ > 0 && time_passed > destroy_after_micros_);
    }

   private:
    // The context for this execution.
    tensorflow::EagerContext* ctx_;

    const WorkerEnv* const env_;  // Not owned.

    mutex last_accessed_mu_;
    int64_t last_accessed_micros_ TF_GUARDED_BY(last_accessed_mu_);
    int64_t destroy_after_micros_;

    const bool is_master_;
  };
  // The returned ServerContext will need to be Unrefed.
  tensorflow::Status GetServerContext(uint64, ServerContext**);

  class ClientTensorHandleDeleteNode : public EagerNode {
   public:
    ClientTensorHandleDeleteNode(
        ServerContext* context,
        std::unique_ptr<RemoteTensorHandleInternal> handle_to_delete)
        : tensorflow::EagerNode(),
          context_(context),
          handle_to_delete_(std::move(handle_to_delete)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_9(mht_9_v, 375, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "ClientTensorHandleDeleteNode");

      context_->Ref();
    }

    ~ClientTensorHandleDeleteNode() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_10(mht_10_v, 382, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "~ClientTensorHandleDeleteNode");
 context_->Unref(); }

    Status Run() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_11(mht_11_v, 387, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "Run");

      VLOG(3) << "ServerContext: Deleting tensor handle "
              << handle_to_delete_->op_id << ":"
              << handle_to_delete_->output_num;
      return context_->Context()->RemoteMgr()->DeleteTensorHandle(
          *handle_to_delete_);
    }

    void Abort(Status status) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_12(mht_12_v, 398, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "Abort");
}

    // Remote node deletions are best effort
    bool Fatal() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_13(mht_13_v, 404, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "Fatal");
 return false; }

    string DebugString() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSeager_service_implDTh mht_14(mht_14_v, 409, "", "./tensorflow/core/distributed_runtime/eager/eager_service_impl.h", "DebugString");

      string out = "[ClientTensorHandleDeleteNode]";
      strings::StrAppend(&out, " op_id: ", handle_to_delete_->op_id);
      strings::StrAppend(&out, ", output_num: ", handle_to_delete_->output_num);
      return out;
    }

   private:
    // Owns one reference.
    ServerContext* const context_;
    const std::unique_ptr<RemoteTensorHandleInternal> handle_to_delete_;
  };

 private:
  Status ExecuteOp(CallOptions* call_opts, const Operation& operation,
                   EagerContext* eager_context, EagerExecutor* eager_executor,
                   QueueResponse* queue_response);
  Status SendTensor(const SendTensorOp& send_tensor,
                    EagerContext* eager_context);
  Status SendPackedHandle(const SendPackedHandleOp& send_packed_handle,
                          EagerContext* eager_context);
  Status RegisterFunction(const RegisterFunctionOp& register_function,
                          EagerContext* eager_context);
  Status CleanupFunction(const CleanupFunctionOp& cleanup_function);
  const WorkerEnv* const env_;  // Not owned.

  mutex contexts_mu_;
  std::unordered_map<uint64, ServerContext*> contexts_
      TF_GUARDED_BY(contexts_mu_);

  std::unique_ptr<Thread> gc_thread_;
  mutex gc_thread_shutdown_mu_;
  condition_variable gc_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(gc_thread_shutdown_mu_) = false;

  TF_DISALLOW_COPY_AND_ASSIGN(EagerServiceImpl);
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_
