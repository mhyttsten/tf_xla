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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh() {
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


#include "grpcpp/alarm.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace tensorflow {
namespace eager {

// This class is a wrapper that handles communication for gRPC.
class GrpcEagerServiceImpl : public AsyncServiceInterface {
 public:
  template <class RequestMessage, class ResponseMessage>
  using EagerCall = Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,
                         RequestMessage, ResponseMessage>;
  template <class RequestMessage, class ResponseMessage>
  using StreamingCall =
      ServerBidirectionalStreamingCall<GrpcEagerServiceImpl,
                                       grpc::EagerService::AsyncService,
                                       RequestMessage, ResponseMessage>;

  GrpcEagerServiceImpl(const WorkerEnv* env,
                       ::grpc::ServerBuilder* server_builder);
  virtual ~GrpcEagerServiceImpl() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h", "~GrpcEagerServiceImpl");
}

  // Create a master context in eager service.
  Status CreateMasterContext(const tensorflow::uint64 context_id,
                             EagerContext* context);

  void HandleRPCsLoop() override;
  void Shutdown() override;

 private:
#define HANDLER(method)                                                       \
  void method##Handler(EagerCall<method##Request, method##Response>* call) {  \
    env_->compute_pool->Schedule([this, call]() {                             \
      call->SendResponse(                                                     \
          ToGrpcStatus(local_impl_.method(&call->request, &call->response))); \
    });                                                                       \
    Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,              \
         method##Request, method##Response>::                                 \
        EnqueueRequest(&service_, cq_.get(),                                  \
                       &grpc::EagerService::AsyncService::Request##method,    \
                       &GrpcEagerServiceImpl::method##Handler, false);        \
  }
  HANDLER(CreateContext);
  HANDLER(UpdateContext);
  HANDLER(WaitQueueDone);
  HANDLER(KeepAlive);
  HANDLER(CloseContext);
#undef HANDLER

  void EnqueueHandler(EagerCall<EnqueueRequest, EnqueueResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      auto call_opts = std::make_shared<CallOptions>();
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      call->SendResponse(ToGrpcStatus(local_impl_.Enqueue(
          call_opts.get(), &call->request, &call->response)));
    });
    Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService, EnqueueRequest,
         EnqueueResponse>::
        EnqueueRequest(&service_, cq_.get(),
                       &grpc::EagerService::AsyncService::RequestEnqueue,
                       &GrpcEagerServiceImpl::EnqueueHandler,
                       /*supports_cancel=*/true);
  }

  void RunComponentFunctionHandler(
      EagerCall<RunComponentFunctionRequest, RunComponentFunctionResponse>*
          call) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh mht_1(mht_1_v, 263, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h", "RunComponentFunctionHandler");

    env_->compute_pool->Schedule([this, call]() {
      auto call_opts = std::make_shared<CallOptions>();
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      local_impl_.RunComponentFunction(call_opts.get(), &call->request,
                                       &call->response,
                                       [call, call_opts](const Status& s) {
                                         call->ClearCancelCallback();
                                         call->SendResponse(ToGrpcStatus(s));
                                       });
    });
    Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,
         RunComponentFunctionRequest, RunComponentFunctionResponse>::
        EnqueueRequest(
            &service_, cq_.get(),
            &grpc::EagerService::AsyncService::RequestRunComponentFunction,
            &GrpcEagerServiceImpl::RunComponentFunctionHandler,
            /*supports_cancel=*/true);
  }

  // Called when a new request has been received as part of a StreamingEnqueue
  // call.
  // StreamingEnqueueHandler gets the request from the `call` and fills the
  // response (also found in `call`) by invoking the local EagerServiceImpl.
  // The local EagerServiceImpl is invoked in a single-threaded thread pool. We
  // do this to preserve request order. The local service can parallelize based
  // on context_id in request if necessary. Remote contexts are created in async
  // mode by default, so the local service impl just puts the request on eager
  // executor queue.
  void StreamingEnqueueHandler(
      StreamingCall<EnqueueRequest, EnqueueResponse>* call) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSeagerPSgrpc_eager_service_implDTh mht_2(mht_2_v, 296, "", "./tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h", "StreamingEnqueueHandler");

    call->Ref();
    enqueue_streaming_thread_.Schedule([this, call]() {
      if (call->RefCountIsOne()) {
        // This StreamingCall has already been shutdown. Don't need to anything.
        call->Unref();
        return;
      }
      // NOTE(fishx): Use the address of StreamingCall as the stream_id since we
      // reuse the same StreamingCall for multiple requests in the same
      // streaming connection.
      Status status = local_impl_.Enqueue(
          /*call_opts=*/nullptr, &call->request(), call->mutable_response(),
          reinterpret_cast<uint64>(static_cast<void*>(call)));

      if (status.ok()) {
        VLOG(1) << "local_impl_.Enqueue completed successfully";
        call->SendResponse();
      } else {
        VLOG(1) << "local_impl_.Enqueue failed with " << status.ToString()
                << " on request " << call->request().DebugString();
        call->Finish(ToGrpcStatus(status));
      }
      call->Unref();

      // We do not tell gRPC to accept a new StreamingEnqueue request because
      // this method can be called multiple times for a given streaming call.
      // The StreamingCall does this per call instead, after a call has been
      // opened.
    });
  }

  const WorkerEnv* const env_;  // Not owned.
  EagerServiceImpl local_impl_;

  // A single-threaded thread pool to handle streaming enqueue rpc request.
  thread::ThreadPool enqueue_streaming_thread_;
  std::unique_ptr<::grpc::Alarm> shutdown_alarm_;

  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::EagerService::AsyncService service_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcEagerServiceImpl);
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_
