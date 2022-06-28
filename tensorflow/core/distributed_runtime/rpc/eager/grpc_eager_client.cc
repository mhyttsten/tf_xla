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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"

#include "grpcpp/generic/generic_stub.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace eager {
namespace {

/*
 * Setting environment variable "TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE" to
 * true will turn on asynchronous execution of remote op. It means that when
 * executing an op on a remote worker, client will not block on waiting
 * for the response anymore. Using follow code as example:
 *
 * with tf.device('worker:0'):
 *   a = tf.matmul(...)
 *   b = tf.matmul(...)
 * logging.into('Requests sent')    # Probably not executed yet
 * logging.info('b: %s', b.numpy()) # Block until 'b' finished.
 *
 * Streaming RPC will preserve order as well. So 'a' must be executed before
 * 'b' on 'worker:0'.
 *
 * When turning on this feature, you should explicitly wait for some result
 * from remote workers at the end of you python program. Otherwise, client may
 * shutdown remote workers without waiting all pending ops.
 *
 * TODO(fishx): When exiting client, make sure all pending ops on remote workers
 * are finished.
 */
bool EnableStreaming() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "EnableStreaming");

  bool result;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE",
                                 true, &result));
  return result;
}

// Ref-counted thread to handle callbacks for completed requests a GRPC
// completion queue. The thread might be shared by multiple eager clients, and
// each one of them should hold a reference count to ensure that the thread
// outlives the clients.
// To ensure that every tag in completion queue is processed, this thread also
// holds a reference to itself and always wait until ref count is one to exit.
class GrpcEagerClientThread : public core::RefCounted {
 public:
  GrpcEagerClientThread() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "GrpcEagerClientThread");

    // Hold a reference to ensure every completion tag gets processed.
    Ref();
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "eager_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcEagerClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcEagerClientThread blocking for next tag";
            if (RefCountIsOne()) {
              break;
            }
          }
          VLOG(4) << "GrpcEagerClientThread exiting";
          completion_queue_.Shutdown();
          // `this` holds the final reference so cannot directly Unref here.
          // Instead, schedule a separate thread to clean it up.
          Env::Default()->SchedClosure([this]() { this->Unref(); });
        }));
  }

  ~GrpcEagerClientThread() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "~GrpcEagerClientThread");
}

  ::grpc::CompletionQueue* completion_queue() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "completion_queue");
 return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class GrpcEagerClient : public EagerClient {
 public:
  GrpcEagerClient(const tensorflow::SharedGrpcChannelPtr& channel,
                  GrpcEagerClientThread* thread, const string& target)
      : stub_(channel), thread_(thread), target_(target) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "GrpcEagerClient");

    // Hold a reference to make sure the corresponding EagerClientThread
    // outlives the client.
    thread_->Ref();
    cq_ = thread->completion_queue();
  }
  ~GrpcEagerClient() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_5(mht_5_v, 299, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "~GrpcEagerClient");
 thread_->Unref(); }

  bool allow_multiple_pending_requests() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_6(mht_6_v, 304, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "allow_multiple_pending_requests");

    return EnableStreaming();
  }

#define CLIENT_METHOD(method)                                             \
  void method##Async(const method##Request* request,                      \
                     method##Response* response, StatusCallback done)     \
      override {                                                          \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));      \
    new RPCState<protobuf::Message>(                                      \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request, \
        response, std::move(done_wrapped), /*call_opts=*/nullptr,         \
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,    \
        &target_);                                                        \
  }

  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(UpdateContext);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);

#undef CLIENT_METHOD

#define CLIENT_CANCELABLE_METHOD(method)                                      \
  void method##Async(CallOptions* call_opts, const method##Request* request,  \
                     method##Response* response, StatusCallback done)         \
      override {                                                              \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));          \
    new RPCState<protobuf::Message>(                                          \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request,     \
        response, std::move(done_wrapped), call_opts, /*threadpool=*/nullptr, \
        /*max_retries=*/0, /*fail_fast=*/true, &target_);                     \
  }

  CLIENT_CANCELABLE_METHOD(Enqueue);
  CLIENT_CANCELABLE_METHOD(RunComponentFunction);

#undef CLIENT_CANCELABLE_METHOD

  void CloseContextAsync(const CloseContextRequest* request,
                         CloseContextResponse* response,
                         StatusCallback done) override {
    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.eager.EagerService/CloseContext", *request,
        response, std::move(done_wrapped), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);

    VLOG(1) << "Sending RPC to close remote eager context "
            << request->DebugString();

    mutex_lock l(mu_);
    const auto& it = enqueue_dispatchers_.find(request->context_id());
    if (it != enqueue_dispatchers_.end()) {
      it->second.CancelCall();
      enqueue_dispatchers_.erase(it);
    } else if (EnableStreaming()) {
      LOG(ERROR) << "Remote EagerContext with id " << request->context_id()
                 << " does not seem to exist.";
    }
  }

  void StreamingEnqueueAsync(CallOptions* call_opts,
                             const EnqueueRequest* request,
                             EnqueueResponse* response,
                             StatusCallback done) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "StreamingEnqueueAsync");

    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    if (EnableStreaming()) {
      mutex_lock l(mu_);
      auto it = enqueue_dispatchers_.find(request->context_id());
      if (it == enqueue_dispatchers_.end()) {
        auto it_and_bool = enqueue_dispatchers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(request->context_id()),
            std::forward_as_tuple(
                &stub_, cq_,
                "/tensorflow.eager.EagerService/StreamingEnqueue"));
        it = it_and_bool.first;
      }
      // TODO(haoyuzhang): Consider supporting cancellation for streaming RPC?
      it->second.SendNextRequest(*request, response, std::move(done_wrapped));
    } else {
      Notification n;
      Status status;
      EnqueueAsync(call_opts, request, response,
                   [&n, &status](const Status& s) {
                     status.Update(s);
                     n.Notify();
                   });
      n.WaitForNotification();
      done_wrapped(status);
    }
  }

 private:
  ::grpc::GenericStub stub_;
  const GrpcEagerClientThread* thread_;
  const string target_;

  ::grpc::CompletionQueue* cq_;

  mutable mutex mu_;

  std::unordered_map<uint64, StreamingRPCDispatcher<EnqueueResponse>>
      enqueue_dispatchers_ TF_GUARDED_BY(mu_);

  StatusCallback callback_wrapper(StatusCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_8(mht_8_v, 417, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "callback_wrapper");

    Ref();
    return [this, done = std::move(done)](const Status& status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_9(mht_9_v, 422, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "lambda");

      done(status);
      this->Unref();
    };
  }
};

class GrpcEagerClientCache : public EagerClientCache {
 public:
  explicit GrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> cache)
      : next_round_robin_assignment_(0), cache_(cache), threads_(4) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_10(mht_10_v, 436, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "GrpcEagerClientCache");

    for (int i = 0, end = threads_.size(); i < end; i++) {
      threads_[i].reset(new GrpcEagerClientThread());
    }
  }

  ~GrpcEagerClientCache() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_11(mht_11_v, 445, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "~GrpcEagerClientCache");
 threads_.clear(); }

  Status GetClient(const string& target,
                   core::RefCountPtr<EagerClient>* client) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_12(mht_12_v, 452, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "GetClient");

    mutex_lock l(clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      tensorflow::SharedGrpcChannelPtr shared =
          cache_->FindWorkerChannel(target);
      if (shared == nullptr) {
        return errors::InvalidArgument("Client for target ", target,
                                       " not found.");
      }
      int assigned_index = AssignClientToThread(target);
      GrpcEagerClientThread* thread = threads_[assigned_index].get();
      core::RefCountPtr<EagerClient> worker(
          new GrpcEagerClient(shared, thread, target));
      it = clients_.emplace(target, std::move(worker)).first;
    }

    it->second->Ref();
    client->reset(it->second.get());
    return Status::OK();
  }

 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const string& target) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_13(mht_13_v, 484, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "AssignClientToThread");

    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  std::shared_ptr<tensorflow::GrpcChannelCache> cache_;
  mutable mutex clients_mu_;
  std::unordered_map<string, core::RefCountPtr<EagerClient>> clients_
      TF_GUARDED_BY(clients_mu_);
  std::vector<core::RefCountPtr<GrpcEagerClientThread>> threads_;
};

}  // namespace

EagerClientCache* NewGrpcEagerClientCache(
    std::shared_ptr<tensorflow::GrpcChannelCache> channel) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_clientDTcc mht_14(mht_14_v, 511, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc", "NewGrpcEagerClientCache");

  return new GrpcEagerClientCache(channel);
}

}  // namespace eager
}  // namespace tensorflow
