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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            thread::ThreadPool* callback_threadpool,
                            WorkerCacheLogger* logger, const string& target)
      : channel_(std::move(channel)),
        stub_(channel_),
        cq_(completion_queue),
        callback_threadpool_(callback_threadpool),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        createworkersession_(Method(GrpcWorkerMethod::kCreateWorkerSession)),
        deleteworkersession_(Method(GrpcWorkerMethod::kDeleteWorkerSession)),
        registergraph_(Method(GrpcWorkerMethod::kRegisterGraph)),
        deregistergraph_(Method(GrpcWorkerMethod::kDeregisterGraph)),
        rungraph_(Method(GrpcWorkerMethod::kRunGraph)),
        cleanupgraph_(Method(GrpcWorkerMethod::kCleanupGraph)),
        cleanupall_(Method(GrpcWorkerMethod::kCleanupAll)),
        recvtensor_(Method(GrpcWorkerMethod::kRecvTensor)),
        recvbuf_(Method(GrpcWorkerMethod::kRecvBuf)),
        logging_(Method(GrpcWorkerMethod::kLogging)),
        tracing_(Method(GrpcWorkerMethod::kTracing)),
        completegroup_(Method(GrpcWorkerMethod::kCompleteGroup)),
        instancesource_(Method(GrpcWorkerMethod::kCompleteInstance)),
        getstepsequence_(Method(GrpcWorkerMethod::kGetStepSequence)),
        markrecvfinished_(Method(GrpcWorkerMethod::kMarkRecvFinished)),
        logger_(logger),
        target_(target) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_0(mht_0_v, 240, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "GrpcRemoteWorker");
}

  ~GrpcRemoteWorker() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "~GrpcRemoteWorker");
}

  void GetStatusAsync(CallOptions* call_opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "GetStatusAsync");

    IssueRequest(request, response, getstatus_, std::move(done), call_opts,
                 fail_fast);
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "CreateWorkerSessionAsync");

    IssueRequest(request, response, createworkersession_, std::move(done));
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "DeleteWorkerSessionAsync");

    IssueRequest(request, response, deleteworkersession_, std::move(done),
                 call_opts);
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "RegisterGraphAsync");

    IssueRequest(request, response, registergraph_, std::move(done));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "DeregisterGraphAsync");

    IssueRequest(request, response, deregistergraph_, std::move(done));
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_7(mht_7_v, 299, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "RunGraphAsync");

    IssueRequest(request, response, rungraph_, std::move(done), call_opts);
  }
  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_8(mht_8_v, 307, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "RunGraphAsync");

    IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                 rungraph_, std::move(done), call_opts);
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_9(mht_9_v, 317, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "CleanupGraphAsync");

    IssueRequest(request, response, cleanupgraph_, std::move(done));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_10(mht_10_v, 326, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "CleanupAllAsync");

    IssueRequest(request, response, cleanupall_, std::move(done));
  }

  void RecvBufAsync(CallOptions* call_opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_11(mht_11_v, 334, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "RecvBufAsync");

    int64_t start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_12(mht_12_v, 343, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "lambda");

      if (logging_active) {
        if (logger_->LoggingActive()) {
          int64_t end_usec = Env::Default()->NowMicros();
          int64_t step_id = request->step_id();
          RecvBufRespExtra extra;
          response->transport_options().UnpackTo(&extra);
          int64_t num_bytes = 0;
          for (const auto& chunk : extra.tensor_content()) {
            num_bytes += chunk.size();
          }
          int64_t send_start_usec = start_usec;
          // Prefer start time reported by the sender, if available.
          if (response->send_start_micros()) {
            send_start_usec =
                std::max(start_usec,
                         static_cast<int64_t>(response->send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->buf_rendezvous_key();
          logger_->RecordDataTransfer(
              step_id, send_start_usec, end_usec, key, request->src_device(),
              request->dst_device(), num_bytes, "", "RecvBuf");
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->DebugString();
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvbuf_, callback, call_opts);
  }

  void CompleteGroupAsync(CallOptions* call_opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_13(mht_13_v, 388, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "CompleteGroupAsync");

    IssueRequest(request, response, completegroup_, std::move(done), call_opts,
                 /*fail_fast=*/false);
  }

  void CompleteInstanceAsync(CallOptions* call_opts,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_14(mht_14_v, 399, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "CompleteInstanceAsync");

    IssueRequest(request, response, instancesource_, std::move(done),
                 call_opts);
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_15(mht_15_v, 409, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "GetStepSequenceAsync");

    IssueRequest(request, response, getstepsequence_, std::move(done));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_16(mht_16_v, 417, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "RecvTensorAsync");

    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64_t start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_17(mht_17_v, 427, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "lambda");

      if (logging_active) {
        if (logger_->LoggingActive()) {
          int64_t end_usec = Env::Default()->NowMicros();
          int64_t step_id = request->step_id();
          int64_t bytes = response->tensor().TotalBytes();
          int64_t send_start_usec = start_usec;
          // If a send start time was reported by the other side, use
          // that instead.  Maybe we should mark the display if we're using
          // our local time instead of the remote start time?
          if (response->metadata().send_start_micros()) {
            // send_start_micros is the timestamp taken when the
            // remote machine began to send the RecvTensor response.
            // Due to clock skew between source and dest machines, it
            // is possible that send_start_micros can be larger than
            // end_usec or less than start_usec.
            //
            // To respect causality, we enforce the invariants that
            // the RecvTensor response can not have been sent before
            // the RecvTensor request, and must have been sent before
            // it was received.
            send_start_usec = std::max(
                start_usec,
                static_cast<int64_t>(response->metadata().send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->rendezvous_key();
          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes);
          }
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->metadata().DebugString();
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->metadata().require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvtensor_, callback, call_opts);
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_18(mht_18_v, 484, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "LoggingAsync");

    IssueRequest(request, response, logging_, done);
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_19(mht_19_v, 492, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "TracingAsync");

    IssueRequest(request, response, tracing_, done);
  }

 private:
  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response, const ::grpc::string& method,
                    StatusCallback done, CallOptions* call_opts = nullptr,
                    bool fail_fast = true) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_20(mht_20_v, 505, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "IssueRequest");

    new RPCState<protobuf::Message>(
        &stub_, cq_, method, *request, response, std::move(done), call_opts,
        callback_threadpool_, MaxRetries(), fail_fast, &target_);
  }

  void IssueRequest(const protobuf::Message* request, TensorResponse* response,
                    const ::grpc::string& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_21(mht_21_v, 516, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "IssueRequest");

    new RPCState<TensorResponse>(&stub_, cq_, method, *request, response,
                                 std::move(done), call_opts,
                                 callback_threadpool_, MaxRetries(),
                                 /*fail_fast=*/true, &target_);
  }

  void IssueMarkRecvFinishedRequest(int64_t request_id) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_22(mht_22_v, 526, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "IssueMarkRecvFinishedRequest");

    VLOG(2) << "Send MarkRecvFinishedRequest for request " << request_id;
    MarkRecvFinishedRequest request;
    request.set_request_id(request_id);

    MarkRecvFinishedResponse* response = new MarkRecvFinishedResponse();
    auto done = [response](Status status) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_23(mht_23_v, 535, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "lambda");
 delete response; };
    IssueRequest(&request, response, markrecvfinished_, done);
  }

  // Helper function for initializing the RpcMethod objects below.
  const char* Method(GrpcWorkerMethod id) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_24(mht_24_v, 543, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "Method");
 return GrpcWorkerMethodName(id); }

  // Helper function for configuring max GRPC retries. Defaults to 0 (no
  // retries).
  const int64_t MaxRetries() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_25(mht_25_v, 550, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "MaxRetries");

    int64_t max_retries = -1;
    TF_CHECK_OK(ReadInt64FromEnvVar("GRPC_MAX_RETRIES", 0, &max_retries));
    return max_retries;
  }

  SharedGrpcChannelPtr channel_;
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  thread::ThreadPool* callback_threadpool_;

  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
  const ::grpc::string deleteworkersession_;
  const ::grpc::string registergraph_;
  const ::grpc::string deregistergraph_;
  const ::grpc::string rungraph_;
  const ::grpc::string cleanupgraph_;
  const ::grpc::string cleanupall_;
  const ::grpc::string recvtensor_;
  const ::grpc::string recvbuf_;
  const ::grpc::string logging_;
  const ::grpc::string tracing_;
  const ::grpc::string completegroup_;
  const ::grpc::string instancesource_;
  const ::grpc::string getstepsequence_;
  const ::grpc::string markrecvfinished_;

  // Support for logging.
  WorkerCacheLogger* logger_;
  const string target_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};

WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     thread::ThreadPool* callback_threadpool,
                                     WorkerCacheLogger* logger,
                                     const string& target) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_workerDTcc mht_26(mht_26_v, 593, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc", "NewGrpcRemoteWorker");

  return new GrpcRemoteWorker(std::move(channel), completion_queue,
                              callback_threadpool, logger, target);
}

}  // namespace tensorflow
