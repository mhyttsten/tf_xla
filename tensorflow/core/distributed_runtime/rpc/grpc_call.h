/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh() {
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


#include "grpcpp/completion_queue.h"
#include "grpcpp/impl/service_type.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/async_stream.h"
#include "grpcpp/support/async_unary_call.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// CALL STRUCTURES
// ===============
//
// Each pending (incoming) request corresponds to a call object that
// encapsulates the state of the call. Templates and
// pointers-to-member functions are used to avoid boilerplate and
// redundant closure creation. The class hierarchy is as follows:
//
// * `UntypedCall<Service>`: The base class represents a call that
//   could be associated with any of the methods on a service of type
//   `Service`. Also defines a `Tag` nested class that can be used as
//   the tag in a `grpc::CompletionQueue`.  Each class that
//   instantiates `Service` should have a completion queue polling
//   loop that knows about `UntypedCall<Service>::Tag` objects, and
//   invokes their `OnCompleted()` method to continue processing.
//
// * `Call<Service, GrpcService, Req, Resp>`: This class extends
//   `UntypedCall<Service>` and is additionally parameterized by the
//   gRPC-generated asynchronous service class, and the request and
//   response message types. It defines the state associated with a
//   call (whose type depends on the message types), and stores a
//   pointer to a `Service::HandleFoo()` handler method. Each
//   `Service::HandleFoo()` method knows about the corresponding
//   `Call` type, in order to access its state, and invoke its
//   `SendResponse()` method.
//
// The lifecycle of a call object is as follows.
//
// 1. A `Service` creates a `Call` for a particular method and
//    enqueues it in its completion queue (via an
//    `UntypedCall<Service>::Tag`).
//
// 2. When the tag is returned from `cq_->Next()`, the
//    `UntypedCall::RequestReceived()` method is invoked and takes
//    ownership of the call object. This indirectly invokes the
//    appropriate handler method on `Service`.
//
// 3. After the response has been written (perhaps in another thread),
//    the `Call::SendResponse()` method is invoked. It transfers
//    ownership of the call object back to the completion queue (via
//    an `UntypedCall::Tag`).
//
// 4. When the response has been sent, the tag is returned from
//    `cq_->Next()`, and the call object is deleted.
//

template <class Service>
class GrpcCallTag {
 public:
  virtual ~GrpcCallTag() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_0(mht_0_v, 249, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "~GrpcCallTag");
}

  // Calls the callback associated with this tag.
  virtual void OnCompleted(Service* service, bool ok) = 0;
};

// Represents a pending request with unknown message types.
template <class Service>
class UntypedCall : public core::RefCounted {
 public:
  virtual ~UntypedCall() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_1(mht_1_v, 262, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "~UntypedCall");
}

  // The implementation of this method should use `service` to handle
  // an incoming request, and (perhaps asynchronously) send the
  // response.
  //
  // One reference on `this` is transferred to the callee, and the
  // callee is responsible for releasing it (typically via
  // `Call::SendResponse()`).
  //
  // `ok` is true if the request was received in a "regular event",
  // otherwise false.
  virtual void RequestReceived(Service* service, bool ok) = 0;

  // This method will be called either (i) when the server is notified
  // that the request has been canceled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  An active Tag owns a reference on the corresponding
  // Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum Callback { kRequestReceived, kResponseSent, kCancelled };

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_2(mht_2_v, 293, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "Tag");
}

    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_3(mht_3_v, 301, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "OnCompleted");

      switch (callback_) {
        case kRequestReceived:
          call_->RequestReceived(service, ok);
          break;
        case kResponseSent:
          // No special handling needed apart from the Unref below.
          break;
        case kCancelled:
          call_->RequestCancelled(service, ok);
          break;
      }
      call_->Unref();  // Ref acquired when tag handed to grpc.
    }

   private:
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_4(mht_4_v, 345, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "Call");
}

  virtual ~Call() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_5(mht_5_v, 350, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "~Call");
}

  void RequestReceived(Service* service, bool ok) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_6(mht_6_v, 355, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "RequestReceived");

    if (ok) {
      this->Ref();
      (service->*handle_request_function_)(this);
    }
  }

  void SendResponse(::grpc::Status status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_7(mht_7_v, 365, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "SendResponse");

    this->Ref();  // Ref for grpc; released in Tag callback.
    responder_.Finish(response, status, &response_sent_tag_);
    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_8(mht_8_v, 374, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "RequestCancelled");

    if (ctx_.IsCancelled()) {
      mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
  }

  // Registers `callback` as the function that should be called if and when this
  // call is canceled by the client.
  void SetCancelCallback(std::function<void()> callback) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_9(mht_9_v, 388, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "SetCancelCallback");

    mutex_lock l(mu_);
    cancel_callback_ = std::move(callback);
  }

  // Clears any cancellation callback that has been registered for this call.
  void ClearCancelCallback() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_10(mht_10_v, 397, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "ClearCancelCallback");

    mutex_lock l(mu_);
    cancel_callback_ = nullptr;
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function,
                             bool supports_cancel) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_11(mht_11_v, 414, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "EnqueueRequest");

    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `method_id`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequestForMethod(
      GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function,
      bool supports_cancel) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_12(mht_12_v, 438, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "EnqueueRequestForMethod");

    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
  }

  RequestMessage request;
  ResponseMessage response;

  const std::multimap<::grpc::string_ref, ::grpc::string_ref>& client_metadata()
      const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_13(mht_13_v, 458, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "client_metadata");

    return ctx_.client_metadata();
  }

 private:
  // Creates a completion queue tag for handling cancellation by the client.
  // NOTE: This method must be called before this call is enqueued on a
  // completion queue.
  void RegisterCancellationHandler() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_14(mht_14_v, 469, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "RegisterCancellationHandler");

    this->Ref();  // Ref for grpc; released in Tag callback.
    ctx_.AsyncNotifyWhenDone(&cancelled_tag_);
  }

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a Call.
  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};
  Tag cancelled_tag_{this, Tag::kCancelled};

  mutex mu_;
  std::function<void()> cancel_callback_ TF_GUARDED_BY(mu_);
};

// Lifetime of a server-side bidirectional streaming call:
// - The call is created in the static EnqueueRequest method. It transfers
//   ownership to the kCallOpen tag pushed onto the completion queue.
// - If kCallOpen completes successfully, a read is requested and the
//   kRequestReceived tag takes ownership of the call. If kCallOpen fails,
//   e.g. server is shutdown, no further requests are pushed and the call is
//   destroyed (at the end of Tag::OnCompleted).
// - When the first request is received, we Ref() the call and invoke the
//   handler method thereby transferring ownership to the handler method.
//   The handler is responsible for calling SendResponse() or Finish() on this
//   call.
//   - If the handler calls Finish(), e.g. the request was invalid, Finish()
//     transfers ownership from the handler to the kServerFinished tag that
//     it pushes on the completion queue. The ownership is transferred because
//     the ref count is not incremented before putting the tag on the queue.
//   - If the handler calls SendResponse(), SendResponse() transfers ownership
//     to the kResponseSent tag.
// - When kResponseSent completes, we request a new read, which owns the call
//   now.
// - When the next request is received, it is handled the same way as the first
//   request.
//
// Because we request a read only after the write is sent, we can safely reuse
// the same request and response messages for the whole call.
template <class Service>
class ServerUntypedBidirectionalStreamingCall : public core::RefCounted {
 public:
  virtual void RequestReceived(Service* service) = 0;

  // Enqueues a request on the completion queue to read the next request.
  virtual void CallOpen() = 0;

  virtual void RequestRead() = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback.
  // An active Tag owns a reference on the corresponding Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum class TagType {
      kCallOpen,
      kRequestReceived,
      kResponseSent,
      kServerFinished,
    };

    Tag(ServerUntypedBidirectionalStreamingCall* call, TagType cb)
        : call_(call), callback_(cb) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_15(mht_15_v, 539, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "Tag");
}

    // Calls the callback associated with this tag and Unrefs this->call_.
    void OnCompleted(Service* service, bool ok) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_16(mht_16_v, 545, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "OnCompleted");

      switch (callback_) {
        case TagType::kCallOpen:
          // Non-ok value indicates that the server has been shutdown before we
          // received a message for this call type. We do nothing to let this
          // call object be destroyed and avoid enqueuing request for another
          // call.
          if (ok) {
            call_->CallOpen();
          }
          break;
        case TagType::kRequestReceived:
          // Non-ok value from completion queue here means that we will not
          // receive any more messages from the client, e.g. the client called
          // WritesDone. There is nothing we need to do in this case. The call
          // will be Unref'ed and deleted. If the client wants to open a new
          // call, we have already enqueued a request for a new call in CallOpen
          // above.
          if (ok) {
            call_->RequestReceived(service);
          }
          break;
        case TagType::kResponseSent:
          if (ok) {
            // The obvious place to request a read would be at the end of
            // RequestReceived(). Unfortunately, this can result in multiple
            // outstanding write requests in the completion queue. This is
            // currently not supported by gRPC, which requires at most one
            // outstanding write request in the completion queue.
            // Requesting a read here, in ResponseSent, works because at
            // this point, the completion queue has no write requests
            // (kResponseSent happens when a write completes).
            // This might be synchronizing the processing more than strictly
            // necessary, but is probably fine because, AFAICT from gRPC docs,
            // the write request completes as soon as it can be written to
            // outgoing buffer.
            call_->RequestRead();
          }
          // ok == false means that the response is not going on the wire
          // because the call is already dead (i.e., canceled, deadline
          // expired, other side dropped the channel, etc). Since the call is
          // dead, there is nothing for us to do, we just let the call be
          // deleted.
          break;
        case TagType::kServerFinished:
          // Whether our finish request is successful or not (whether it went
          // on the wire towards the client), there is nothing for us to do.
          // In the current implementation, there can be no read or write
          // requests in the completion queue (see the comment in kResponseSent)
          // above. Even if there were pending requests, they would complete
          // with a non-ok status, we would not do anything, and let the call be
          // deleted.
          break;
      }
      call_->Unref();  // Ref acquired when tag was handed to grpc.
    }

   private:
    ServerUntypedBidirectionalStreamingCall* const
        call_;  // `this` owns one reference.
    TagType callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
// Common usage pattern is to have a single thread waiting on events from
// completion queue and calling Tag::OnCompleted(), which invokes methods
// on this.
// This implementation assumes that the server will generate a single response
// message for each request message. More precisely, this class expects that
// each time it invokes handle_request_function_, the service implementation
// will either call SendResponse or Finish exactly once.
// Not thread-safe.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class ServerBidirectionalStreamingCall
    : public ServerUntypedBidirectionalStreamingCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*,
      ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      ServerBidirectionalStreamingCall<Service, GrpcService, RequestMessage,
                                       ResponseMessage>*);

  ServerBidirectionalStreamingCall(
      HandleRequestFunction handle_request_function, GrpcService* grpc_service,
      ::grpc::ServerCompletionQueue* cq, EnqueueFunction enqueue_function)
      : handle_request_function_(handle_request_function),
        stream_(&ctx_),
        grpc_service_(grpc_service),
        cq_(cq),
        enqueue_function_(enqueue_function) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_17(mht_17_v, 648, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "ServerBidirectionalStreamingCall");

    VLOG(3) << "Creating ServerBidirectionalStreamingCall " << this;
  }

  ~ServerBidirectionalStreamingCall() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_18(mht_18_v, 655, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "~ServerBidirectionalStreamingCall");

    VLOG(3) << "Destroying ServerBidirectionalStreamingCall " << this;
  }

  void CallOpen() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_19(mht_19_v, 662, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "CallOpen");

    // Let gRPC know that we can accept another call.
    ServerBidirectionalStreamingCall<
        Service, GrpcService, RequestMessage,
        ResponseMessage>::EnqueueRequest(grpc_service_, cq_, enqueue_function_,
                                         handle_request_function_);
    RequestRead();
  }

  void RequestRead() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_20(mht_20_v, 674, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "RequestRead");

    this->Ref();
    request_.Clear();
    stream_.Read(&request_, &request_received_tag_);
  }

  void RequestReceived(Service* service) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_21(mht_21_v, 683, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "RequestReceived");

    this->Ref();
    // Request handling should result in a call to SendResponse or Finish.
    (service->*handle_request_function_)(this);
  }

  void SendResponse() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_22(mht_22_v, 692, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "SendResponse");

    // Transferring ownership of this to the response_sent_tag_.
    stream_.Write(response_, &response_sent_tag_);
    // stream_.Write does not save references to response_. We are free to muck
    // around with it as soon as Write returns.
    // We clear the response_ to prepare it for the next response.
    response_.Clear();
  }

  void Finish(::grpc::Status status) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_23(mht_23_v, 704, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "Finish");

    // Transferring ownership of this to the server_finished_tag_.
    stream_.Finish(status, &server_finished_tag_);
  }

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled by the given `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_24(mht_24_v, 719, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "EnqueueRequest");

    auto call =
        new ServerBidirectionalStreamingCall<Service, GrpcService,
                                             RequestMessage, ResponseMessage>(
            handle_request_function, grpc_service, cq, enqueue_function);

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->stream_, cq, cq,
                                      &call->call_open_tag_);
  }

  const RequestMessage& request() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_25(mht_25_v, 733, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "request");
 return request_; }
  ResponseMessage* mutable_response() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_callDTh mht_26(mht_26_v, 737, "", "./tensorflow/core/distributed_runtime/rpc/grpc_call.h", "mutable_response");
 return &response_; }

 private:
  // Request and response messages are reused for each request/response exchange
  // between the client and the server.
  RequestMessage request_;
  ResponseMessage response_;
  ::grpc::ServerContext ctx_;

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage> stream_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a ServerBidirectionalStreamingCall.
  typedef typename ServerUntypedBidirectionalStreamingCall<Service>::Tag Tag;
  // At most one tag of each kind may be given to gRPC at any one time.
  // Beyond semantic sanity, this is needed to ensure proper ref counting
  // of this call object.
  Tag call_open_tag_{this, Tag::TagType::kCallOpen};
  Tag request_received_tag_{this, Tag::TagType::kRequestReceived};
  Tag response_sent_tag_{this, Tag::TagType::kResponseSent};
  Tag server_finished_tag_{this, Tag::TagType::kServerFinished};

  // These fields are used only to spawn another instance of this to accept
  // more streaming calls.
  GrpcService* grpc_service_;
  ::grpc::ServerCompletionQueue* cq_;
  EnqueueFunction enqueue_function_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
