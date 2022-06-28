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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh() {
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


#include <queue>
#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// Object allocated per active RPC.
// Manage the state of a single asynchronous RPC request.  If `max_retries`
// is greater than 0, the request will be retried for any transient failures.
template <class Response>
class RPCState : public GrpcClientCQTag {
 public:
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const protobuf::Message& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool, int32_t max_retries = 0,
           bool fail_fast = true, const string* target = nullptr)
      : RPCState(
            stub, cq, method, request, response, std::move(done), call_opts,
            threadpool,
            // 1) If GRPC_FAIL_FAST is set to 'true' or 'false',
            // fail_fast=$GRPC_FAIL_FAST. See b/141948186.
            // 2) Otherwise if GRPC_FAIL_FAST is set to 'use_caller', use the
            // fail_fast from the caller. See b/140260119.
            //
            // Current default: use caller's fail_fast argument.
            //
            // NOTE: Callers mostly set fail_fast=true to prevent job hanging
            // on worker task failures, except a few cases such as GetStatus
            // in cluster initialization and collective param resolution.
            [fail_fast, &done]() -> bool {
              string fail_fast_env;
              TF_CHECK_OK(ReadStringFromEnvVar("GRPC_FAIL_FAST", "use_caller",
                                               &fail_fast_env));
              string fail_fast_env_lower = absl::AsciiStrToLower(fail_fast_env);
              if (fail_fast_env_lower == "true") {
                return true;
              } else if (fail_fast_env_lower == "use_caller") {
                return fail_fast;
              } else if (fail_fast_env_lower == "false") {
                return false;
              } else {
                string error_message = strings::StrCat(
                    "Invalid GRPC_FAIL_FAST config: ", fail_fast_env);
                LOG(WARNING) << error_message;
                done(errors::InvalidArgument(error_message));
                return false;
              }
            }(),
            (call_opts != nullptr ? call_opts->GetTimeout() : 0), max_retries,
            target) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "RPCState");
}

  template <typename Request>
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const Request& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool, bool fail_fast,
           int64_t timeout_in_ms, int32_t max_retries, const string* target)
      : call_opts_(call_opts),
        threadpool_(threadpool),
        done_(std::move(done)),
        timeout_in_ms_(timeout_in_ms),
        max_retries_(max_retries),
        cq_(cq),
        stub_(stub),
        method_(method),
        fail_fast_(fail_fast),
        target_(target) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_1(mht_1_v, 272, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "RPCState");

    response_ = response;
    ::grpc::Status s = GrpcMaybeUnparseProto(request, &request_buf_);
    if (!s.ok()) {
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << s.error_message();
      // Skip retry logic if we fail to parse our request.
      done_(FromGrpcStatus(s));
      delete this;
      return;
    }
    StartCall();
  }

  void StartCall() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_2(mht_2_v, 289, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "StartCall");

    context_.reset(new ::grpc::ClientContext());
    context_->set_wait_for_ready(!fail_fast_);
    if (timeout_in_ms_ > 0) {
      context_->set_deadline(
          gpr_time_from_millis(timeout_in_ms_, GPR_TIMESPAN));
    }
    if (call_opts_) {
      call_opts_->SetCancelCallback([this]() { context_->TryCancel(); });
    }

    VLOG(2) << "Starting call: " << method_;

    call_ = stub_->PrepareUnaryCall(context_.get(), method_, request_buf_, cq_);
    call_->StartCall();
    call_->Finish(&response_buf_, &status_, this);
  }

  void OnCompleted(bool ok) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_3(mht_3_v, 310, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "OnCompleted");

    if (call_opts_) {
      call_opts_->ClearCancelCallback();
    }

    VLOG(2) << "Completed call: " << method_;

    Status s = FromGrpcStatus(status_);
    if (s.ok() && !ok) {
      // Since this function is only being used for processing the response
      // to Finish for client-side unary calls, ok should never be false
      s.Update(
          errors::Internal("GRPC status is okay but CompletionQueueStatus is "
                           "not.  This should never happen."));
    }

    if (s.ok()) {
      if (threadpool_) {
        // Run parse and callback in another thread, returning this
        // one to service more RPCs.
        threadpool_->Schedule([this]() { ParseAndCallDone(); });
      } else {
        ParseAndCallDone();
      }
      return;
    }

    VLOG(1) << method_ << " returned with non-ok status: " << s
            << " Retries: " << num_retries_ << " Max: " << max_retries_ << "\n"
            << context_->debug_error_string();
    // Retry if we have any attempts left
    if (++num_retries_ <= max_retries_ &&
        (errors::IsUnavailable(s) || errors::IsUnknown(s))) {
      response_buf_.Clear();
      VLOG(1) << "Retrying call for " << method_ << "Retry: " << num_retries_
              << " of " << max_retries_;

      ComputeRetryBackoffMs(/*min_backoff_ms=*/1, /*max_backoff_ms=*/10000);
      int64_t backoff_us = retry_backoff_ms_ * 1000;
      Env::Default()->SchedClosureAfter(/*micros=*/backoff_us,
                                        [this]() { StartCall(); });
    } else {
      // Attach additional GRPC error information if any to the final status
      string error_msg = s.error_message();
      strings::StrAppend(&error_msg, "\nAdditional GRPC error information");
      if (target_) {
        strings::StrAppend(&error_msg, " from remote target ", *target_);
      }
      strings::StrAppend(&error_msg, ":\n:", context_->debug_error_string());
      s = errors::CreateWithUpdatedMessage(s, error_msg);
      // Always treat gRPC cancellation as a derived error. This ensures that
      // other error types are preferred during status aggregation. (gRPC
      // cancellation messages do not contain the original status message).
      if (s.code() == tensorflow::error::Code::CANCELLED) {
        s = StatusGroup::MakeDerived(s);
      }

      done_(s);
      delete this;
    }
  }

  void ParseAndCallDone() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_4(mht_4_v, 375, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "ParseAndCallDone");

    Status s;
    if (!GrpcMaybeParseProto(&response_buf_, response_)) {
      s.Update(errors::Internal("could not parse rpc response"));
    }
    done_(s);
    delete this;
  }

 private:
  void ComputeRetryBackoffMs(int min_backoff_ms, int max_backoff_ms) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_5(mht_5_v, 388, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "ComputeRetryBackoffMs");

    constexpr float kBackoffBase = 1.3;
    if (retry_backoff_ms_ < 0) {
      retry_backoff_ms_ = min_backoff_ms;
    } else {
      retry_backoff_ms_ *= kBackoffBase;
      if (retry_backoff_ms_ > max_backoff_ms) {
        retry_backoff_ms_ = max_backoff_ms;
      }
    }
  }

  CallOptions* call_opts_;
  std::unique_ptr<::grpc::ClientContext> context_;
  thread::ThreadPool* threadpool_;
  std::unique_ptr<::grpc::GenericClientAsyncResponseReader> call_;
  Response* response_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  ::grpc::Status status_;
  StatusCallback done_;
  int64_t timeout_in_ms_;

  size_t num_retries_ = 0;
  size_t max_retries_;
  double retry_backoff_ms_ = -1;

  ::grpc::CompletionQueue* cq_;
  ::grpc::GenericStub* stub_;
  ::grpc::string method_;
  bool fail_fast_;
  const string* target_;
};

// Represents state associated with one streaming RPC call.
// Similarly to above, we extract the methods of StreamingRPCState that don't
// need to be templated into this abstract class.
// Currently, *StreamingRPCState does not support client closing the call as
// there is no use case for it - current clients keep the streaming call open
// as long as possible. If/when the need arises, support can be added
// by calling GenericClientAsyncReaderWriter::WritesDone with a new tag
// TagType::kClientFinished and handling the completion in a new callback.
class UntypedStreamingRPCState : public core::RefCounted {
 public:
  virtual void CallStarted(bool ok) = 0;
  virtual void RequestWriteCompleted(bool ok) = 0;
  virtual void ResponseReadCompleted(bool ok) = 0;
  virtual void CallFinished(bool ok) = 0;

  virtual string DebugString() const = 0;

  class Tag : public GrpcClientCQTag {
   public:
    // One enum value per supported callback.
    enum class TagType {
      kCallStarted,
      kRequestWriteCompleted,
      kResponseReadCompleted,
      kCallFinished,
    };

    Tag(UntypedStreamingRPCState* streaming_state, Tag::TagType type);

    // Calls the callback associated with this tag and Unrefs
    // `this->streaming_state_`.
    void OnCompleted(bool ok) override;

   private:
    // OnCompleted() consumes on reference each time it is called.
    UntypedStreamingRPCState* const streaming_state_;
    const Tag::TagType type_;
  };
};

const char* ToString(UntypedStreamingRPCState::Tag::TagType tag_type);

// Represents a single request/response exchange between client and the server.
// A single streaming call contains a sequence of exchanges. Besides the
// messages, exchange contains:
//  - the user callback to invoke when exchange completes (response is received
//    or an error occurs).
//  - The current state of the exchange.
class Exchange {
 public:
  enum class State {
    kExchangeCreated,
    kRequestWriteIssued,
    kRequestWriteCompleted,
    kResponseReadIssued,
  };

  Exchange(const ::grpc::ByteBuffer& request_buf, protobuf::Message* response,
           StatusCallback cb, string debug_string)
      : state_(State::kExchangeCreated),
        request_buf_(request_buf),
        response_(response),
        cb_(std::move(cb)),
        debug_string_(std::move(debug_string)) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("debug_string: \"" + debug_string + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_6(mht_6_v, 489, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "Exchange");
}

  const ::grpc::ByteBuffer& request_buf() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_7(mht_7_v, 494, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "request_buf");
 return request_buf_; }
  ::grpc::ByteBuffer* response_buf() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_8(mht_8_v, 498, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "response_buf");
 return &response_buf_; }

  void MarkRequestWriteIssued() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_9(mht_9_v, 503, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MarkRequestWriteIssued");

    DCHECK(state_ == State::kExchangeCreated);
    state_ = State::kRequestWriteIssued;
  }
  void MarkRequestWriteCompleted() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_10(mht_10_v, 510, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MarkRequestWriteCompleted");

    DCHECK(state_ == State::kRequestWriteIssued);
    state_ = State::kRequestWriteCompleted;
  }
  void MarkResponseReadIssued() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_11(mht_11_v, 517, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MarkResponseReadIssued");

    DCHECK(state_ == State::kRequestWriteCompleted);
    state_ = State::kResponseReadIssued;
  }

  // If `status` is success, completes this exchange by parsing the
  // response_buf_ and invoking cb_ with Status::OK(). Else, invokes the
  // callback with `status`.
  void Complete(Status status);

  const State& state() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_12(mht_12_v, 530, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "state");
 return state_; }

  string DebugString() const;

 private:
  State state_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  protobuf::Message* response_;
  StatusCallback cb_;
  string debug_string_;
};

const char* ToString(Exchange::State s);

std::ostream& operator<<(std::ostream& os, const Exchange::State& state);

// Represents a queue of exchanges.
// When a client sends a new request a new exchange is created and added to the
// end of the queue. Completed exchanges are popped from the front of the queue.
// An explicit exchange queue is needed to brdige the client, which can send new
// requests at any time, with gRPC infrastructure, which can handle a single
// read and a single write request at a time.
//
// As the exchange progresses (request sending initiated, request sending
// completed, response reading initiated) the queue helps to make sure that the
// right operation is issued on the right exchange at the right time.
//
// To satisfy gRPC constraints, the states of exchanges must be as follows
// starting from the front of the queue:
//  - 0 or 1 exchange in kResponseReadIssued state
//  - 0 or more exchanges in kRequestWriteCompleted state
//  - 0 or 1 exchange in kRequestWriteIssued state
//  - 0 or more exchanges in kExchangeCreated state
//
// Thread-compatible.
class ExchangeQueue {
 public:
  // Creates a new exchange and adds it to the end of the queue.
  void Emplace(const ::grpc::ByteBuffer& request_buf,
               protobuf::Message* response, StatusCallback cb,
               std::string debug_string);

  // Returns an exchange for which we can initiate request writing, if any.
  // Returns nullptr if there is no such exchange.
  Exchange* GetReadyForRequestWriting();

  // Returns an exchange for which we can initiate response reading, if any.
  // Returns nullptr if there is no such exchange.
  Exchange* GetReadyForResponseReading();

  // Changes the state of the exchange that is current in kRequestWriteIssued
  // state to kRequestWriteCompleted state.
  // REQUIRES: There is an exchange in kRequestWriteIssued state.
  void MarkRequestWriteCompleted();

  // Returns the exchange at the front of the queue.
  // REQUIRES: ExchangeQueue is not empty.
  Exchange& GetFront();

  // Removes the exchange at the front of the queue.
  // REQUIRES: ExchangeQueue is not empty.
  void PopFront();

  // Returns a string containing addresses and states of all exchanges in this
  // queue.
  string DebugString() const;

  // Swaps the contents of this and `other`.
  void Swap(ExchangeQueue* other);

  // Completes all exchanges in this with `status`.
  void CompleteAll(Status status);

  void CallStarted() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_13(mht_13_v, 607, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "CallStarted");
 call_started_ = true; }

 private:
  // Does nothing by default. Turn on VLOG(5) to enable.
  // Checks that this ExchangeQueue is in a valid state.
  // Kills the process if not.
  void CheckInvariants();

  // We can't process any exchanges until the call has started.
  bool call_started_ = false;

  // std::queue is based on std::deque by default. std::deque provides
  // fairly strong iterator stability.
  std::deque<Exchange> exchanges_;
};  // namespace tensorflow

// Represents state associated with one streaming RPC call.
// Thread-safe
template <class Response>
class StreamingRPCState : public UntypedStreamingRPCState {
 public:
  // Default behavior is to set fail_fast = False and handle timeouts
  // manually.
  StreamingRPCState(
      std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call,
      const std::shared_ptr<::grpc::ClientContext>& context)
      : context_(context), call_(std::move(call)), call_state_(State::kActive) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_14(mht_14_v, 636, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "StreamingRPCState");

    Ref();
    VLOG(3) << "Created new StreamingRPCState " << this;
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::StartCall";
    call_->StartCall(&call_started_tag_);
  }

  ~StreamingRPCState() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_15(mht_15_v, 646, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "~StreamingRPCState");

    VLOG(3) << "Destructing StreamingRPCState " << this;
  }

  // Attempts to send the next request. `done` is invoked when
  // `response` has been filled with the data from the server, or if there
  // is an error. `done` can be invoked before SendNextRequest returns.
  // Return `true` if the call is alive and the `done` callback has or
  // will be invoked. If the call is dead, returns `false`. `done` callback
  // will not be invoked in this case.
  // REQUIRES: The call has been started, i.e. WaitForCallStarted() has
  // returned.
  bool SendNextRequest(const protobuf::Message& request, Response* response,
                       const StatusCallback& done) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_16(mht_16_v, 662, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "SendNextRequest");

    ::grpc::ByteBuffer request_buf;
    ::grpc::Status s = GrpcMaybeUnparseProto(request, &request_buf);
    if (!s.ok()) {
      Status status = FromGrpcStatus(s);
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << status.ToString();
      done(status);
      return true;
    }

    mutex_lock l(mu_);
    if (call_state_ != State::kActive) {
      // `done` is not invoked intentionally.
      return false;
    }
    if (VLOG_IS_ON(3)) {
      // If vlog 3 is enabled, include first 100 chars of request as debug
      // string.
      exchanges_.Emplace(request_buf, response, done,
                         request.ShortDebugString().substr(0, 100));
    } else {
      exchanges_.Emplace(request_buf, response, done, "");
    }
    MaybeIssueRequestWriteLocked();
    return true;
  }

  void CallStarted(bool ok) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_17(mht_17_v, 693, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "CallStarted");

    VLOG(3) << "StreamingRPCState(" << this << ")::CallStarted(ok=" << ok
            << ")";
    mutex_lock l(mu_);
    if (!ok) {
      call_state_ = State::kDone;
      return;
    }
    exchanges_.CallStarted();
    // Now that the call has started, we can write our first request, if any.
    MaybeIssueRequestWriteLocked();
  }

  void RequestWriteCompleted(bool ok) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_18(mht_18_v, 709, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "RequestWriteCompleted");

    VLOG(3) << "StreamingRPCState(" << this
            << ")::RequestWriteCompleted(ok=" << ok << ")";
    mu_.lock();
    if (call_state_ != State::kActive) {
      mu_.unlock();
      return;
    }
    exchanges_.MarkRequestWriteCompleted();
    // Issue ResponseRead regardless of OK status on completing RequestWrite.
    // If the underlying completion queue is in Not-OK status due to previous
    // request failuress (i.e., `ok` from `Next` call on completion queue is
    // False), delay the error in ResponseRead so we can get the remote error
    // message from response buffer.
    MaybeIssueResponseReadLocked();

    if (ok) {
      MaybeIssueRequestWriteLocked();
    }
    mu_.unlock();
  }

  void ResponseReadCompleted(bool ok) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_19(mht_19_v, 734, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "ResponseReadCompleted");

    VLOG(3) << "StreamingRPCState(" << this
            << ")::ResponseReadCompleted(ok=" << ok << ")";
    mu_.lock();
    if (call_state_ != State::kActive) {
      mu_.unlock();
      return;
    }
    if (!ok) {
      IssueCallFinishLocked();
      mu_.unlock();
      return;
    }

    // Complete the exchange without holding the lock because user's
    // callback can call back into this RPC code resulting in a deadlock.
    // No other thread can pop this exchange while we release the lock because
    // this is the only method that pops exchanges and it is called from a
    // single thread that waits on completion queue events.
    Exchange* e;
    e = &exchanges_.GetFront();
    mu_.unlock();

    e->Complete(Status::OK());

    {
      mutex_lock l(mu_);
      exchanges_.PopFront();
      MaybeIssueResponseReadLocked();
    }
  }

  void CallFinished(bool ok) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_20(mht_20_v, 769, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "CallFinished");

    VLOG(3) << "StreamingRPCState(" << this << ")::CallFinished(ok=" << ok
            << ")";
    mu_.lock();
    DCHECK(call_state_ != State::kActive);
    if (call_state_ != State::kFinishing) {
      mu_.unlock();
      return;
    }

    Status s = FromGrpcStatus(call_status_);
    if (s.ok() && !ok) {
      s.Update(
          errors::Internal("GRPC status is okay but CompletionQueueStatus is "
                           "not.  This should never happen.",
                           context_->debug_error_string()));
    }
    // unlocks mu_
    MarkDoneAndCompleteExchanges(s);
  }

  string DebugString() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_21(mht_21_v, 793, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "DebugString");

    mutex_lock l(mu_);
    return exchanges_.DebugString();
  }

 private:
  enum class State {
    kActive,
    kFinishing,
    kDone,
  };

  void MarkDoneAndCompleteExchanges(Status status)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_UNLOCK_FUNCTION(mu_) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_22(mht_22_v, 809, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MarkDoneAndCompleteExchanges");

    call_state_ = State::kDone;
    VLOG(2) << "Ending gRPC streaming call on the client side due to "
            << status.ToString();
    // Swap the exchanges_ into a temporary ExchangeQueue so that we can
    // complete all exchanges without holding mu_ in case user callback
    // reach back into this. This should be impossible now, but safer for
    // the future.
    ExchangeQueue queue;
    exchanges_.Swap(&queue);
    mu_.unlock();
    queue.CompleteAll(status);
  }

  void MaybeIssueRequestWriteLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_23(mht_23_v, 826, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MaybeIssueRequestWriteLocked");

    Exchange* exchange = exchanges_.GetReadyForRequestWriting();
    if (exchange == nullptr) {
      // There are no queued exchanges, there is already an outstanding write,
      // or there are no just created exchanges.
      return;
    }
    exchange->MarkRequestWriteIssued();
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Write";
    call_->Write(exchange->request_buf(), &request_write_completed_tag_);
  }

  void MaybeIssueResponseReadLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_24(mht_24_v, 842, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "MaybeIssueResponseReadLocked");

    Exchange* exchange = exchanges_.GetReadyForResponseReading();
    if (exchange == nullptr) {
      return;
    }
    exchange->MarkResponseReadIssued();
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Read";
    call_->Read(exchange->response_buf(), &response_read_completed_tag_);
  }

  void IssueCallFinishLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_25(mht_25_v, 856, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "IssueCallFinishLocked");

    call_state_ = State::kFinishing;
    Ref();
    VLOG(3) << "StreamingRPCState(" << this << ") calling grpc::Finish";
    // We call finish in response to completed (with error) response reading tag
    // on some exchange. We let this exchange hang in ResponseReadIssued state.
    // ExchangeQueue makes sure that there is at most one exchange in this
    // state. So, no new reads will be issued.
    call_->Finish(&call_status_, &finished_tag_);
  }

  // Holds state for a single request/response exchange between the client
  // and the server.
  typedef typename UntypedStreamingRPCState::Tag Tag;

  // Order of context_ and call_ is important because context_ must outlive
  // call_.
  const std::shared_ptr<const ::grpc::ClientContext> context_;
  std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call_;

  mutable mutex mu_;
  ExchangeQueue exchanges_ TF_GUARDED_BY(mu_);
  State call_state_ TF_GUARDED_BY(mu_);
  ::grpc::Status call_status_ TF_GUARDED_BY(mu_);

  // We can get away with having single instances of these tags per
  // StreamingRPCState because we make sure (as gRPC requires) that
  // there is at most one outstanding Read and at most one outstanding Write
  // in the completion queue.
  // Tags are immutable. No need to guard them.
  Tag call_started_tag_{this, Tag::TagType::kCallStarted};
  Tag request_write_completed_tag_{this, Tag::TagType::kRequestWriteCompleted};
  Tag response_read_completed_tag_{this, Tag::TagType::kResponseReadCompleted};
  Tag finished_tag_{this, Tag::TagType::kCallFinished};
};

// Creates streaming calls and dispatches requests to them.
// In the common case, the client would create a StreamingRPCDispatcher for
// each bidirectional streaming RPC it might want to make. The first time, it
// calls SendNextRequest, a streaming call is initiated and the request is
// sent within this call. Initiation of the call blocks the client. If there are
// no errors, subsequent calls to SendNextRequest would use the already active
// call. If there was an error, the call object will be destroyed after all
// the callbacks for outstanding requests have been invoked. The next call to
// SendNextRequest will initiate a new call.
//
// Callbacks that are part of the same call, are invoked in the order they were
// provided, but callbacks across calls (a failed and a new one) can be invoked
// in any order.
//
// Thread-safe.
template <class Response>
class StreamingRPCDispatcher {
 public:
  StreamingRPCDispatcher(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
                         const ::grpc::string& method)
      : stub_(stub), cq_(cq), method_(method) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_26(mht_26_v, 915, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "StreamingRPCDispatcher");
}

  // Attempts to send the next request. If there is no active streaming call,
  // starts one and sends the request on top of it. `done` is invoked when
  // `response` has been filled with the data from the server, or if there
  // is an error. `done` can be invoked before SendNextRequest returns.
  void SendNextRequest(const protobuf::Message& request, Response* response,
                       StatusCallback done) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_27(mht_27_v, 925, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "SendNextRequest");

    mutex_lock l(mu_);
    if (state_ == nullptr) {
      CreateStreamingState();
    }

    bool is_call_alive = state_->SendNextRequest(request, response, done);
    if (is_call_alive) {
      return;
    }

    // The attempt to send failed because the call was dead, create a new
    // call and try again. When the call is dead SendNextRequest does not call
    // `done`.
    CreateStreamingState();

    is_call_alive = state_->SendNextRequest(request, response, done);
    if (!is_call_alive) {
      // Consider retrying to create and start a call few more times.
      done(errors::Unknown("gRPC call failed right after it was created"));
    }
  }

  // Request to cancel the current streaming call. Non-blocking.
  void CancelCall() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_28(mht_28_v, 952, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "CancelCall");

    mutex_lock l(mu_);
    if (state_ == nullptr) {
      return;
    }
    context_->TryCancel();
    state_ = nullptr;
  }

 private:
  void CreateStreamingState() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_stateDTh mht_29(mht_29_v, 965, "", "./tensorflow/core/distributed_runtime/rpc/grpc_state.h", "CreateStreamingState");

    // ClientContext cannot be reused across calls.
    context_ = std::make_shared<::grpc::ClientContext>();
    // Don't immediately fail StartCall if the channel is not ready. Wait for
    // the channel to become ready.
    context_->set_wait_for_ready(true);

    std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call =
        stub_->PrepareCall(context_.get(), method_, cq_);

    state_.reset(new StreamingRPCState<Response>(std::move(call), context_));
  }

  mutable mutex mu_;

  // Both are thread-safe
  ::grpc::GenericStub* const stub_;
  ::grpc::CompletionQueue* const cq_;

  // Does not need synchronization since it is constant.
  const ::grpc::string method_;

  std::shared_ptr<::grpc::ClientContext> context_ TF_GUARDED_BY(mu_);
  core::RefCountPtr<StreamingRPCState<Response>> state_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
