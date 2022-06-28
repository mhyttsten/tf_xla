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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc() {
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

// GrpcMasterService implements the RPC service MasterService.
//
// A GrpcMasterService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcMasterService knows ahead of time local devices available as
// client devices.
//
// A GrpcMasterService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"

#include "grpcpp/alarm.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, const ConfigProto& default_session_config,
                    ::grpc::ServerBuilder* builder)
      : master_impl_(master),
        is_shutdown_(false),
        default_session_config_(default_session_config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "GrpcMasterService");

    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
  }

  ~GrpcMasterService() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "~GrpcMasterService");
 delete shutdown_alarm_; }

  void Shutdown() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "Shutdown");

    bool did_shutdown = false;
    {
      mutex_lock l(mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcMasterService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      // NOTE(mrry): This enqueues a special event (with a null tag)
      // that causes the completion queue to be shut down on the
      // polling thread.
      shutdown_alarm_ =
          new ::grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(RunStep);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                              \
  do {                                                                        \
    mutex_lock l(mu_);                                                        \
    if (!is_shutdown_) {                                                      \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,              \
           method##Request, method##Response>::                               \
          EnqueueRequest(&master_service_, cq_.get(),                         \
                         &grpc::MasterService::AsyncService::Request##method, \
                         &GrpcMasterService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)

  void HandleRPCsLoop() override {
    ENQUEUE_REQUEST(CreateSession, true);
    ENQUEUE_REQUEST(ExtendSession, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(PartialRunSetup, false);
      ENQUEUE_REQUEST(RunStep, true);
    }
    ENQUEUE_REQUEST(CloseSession, false);
    ENQUEUE_REQUEST(ListDevices, false);
    ENQUEUE_REQUEST(Reset, false);
    ENQUEUE_REQUEST(MakeCallable, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(RunCallable, true);
    }
    ENQUEUE_REQUEST(ReleaseCallable, false);

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      UntypedCall<GrpcMasterService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcMasterService>::Tag*>(tag);
      if (callback_tag) {
        callback_tag->OnCompleted(this, ok);
      } else {
        // NOTE(mrry): A null `callback_tag` indicates that this is
        // the shutdown alarm.
        cq_->Shutdown();
      }
    }
  }

 private:
  Master* master_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  mutex mu_;
  bool is_shutdown_ TF_GUARDED_BY(mu_);
  const ConfigProto default_session_config_;
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template <class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  // RPC handler for creating a session.
  void CreateSessionHandler(
      MasterCall<CreateSessionRequest, CreateSessionResponse>* call) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_3(mht_3_v, 329, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "CreateSessionHandler");

    CreateSessionRequest* rewritten_req = new CreateSessionRequest;
    rewritten_req->mutable_config()->MergeFrom(default_session_config_);
    rewritten_req->MergeFrom(call->request);
    master_impl_->CreateSession(rewritten_req, &call->response,
                                [call, rewritten_req](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                  delete rewritten_req;
                                });
    ENQUEUE_REQUEST(CreateSession, true);
  }

  // RPC handler for extending a session.
  void ExtendSessionHandler(
      MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_4(mht_4_v, 346, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "ExtendSessionHandler");

    master_impl_->ExtendSession(&call->request, &call->response,
                                [call](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                });
    ENQUEUE_REQUEST(ExtendSession, false);
  }

  // RPC handler for setting up a partial run call.
  void PartialRunSetupHandler(
      MasterCall<PartialRunSetupRequest, PartialRunSetupResponse>* call) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_5(mht_5_v, 359, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "PartialRunSetupHandler");

    master_impl_->PartialRunSetup(&call->request, &call->response,
                                  [call](const Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(PartialRunSetup, false);
  }

  // RPC handler for running one step in a session.
  void RunStepHandler(MasterCall<RunStepRequest, RunStepResponse>* call) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_6(mht_6_v, 371, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "RunStepHandler");

    auto* trace = TraceRpc("RunStep/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    if (call->request.options().timeout_in_ms() > 0) {
      call_opts->SetTimeout(call->request.options().timeout_in_ms());
    } else {
      call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    }
    RunStepRequestWrapper* wrapped_request =
        new ProtoRunStepRequest(&call->request);
    MutableRunStepResponseWrapper* wrapped_response =
        new NonOwnedProtoRunStepResponse(&call->response);
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunStep(
        call_opts, wrapped_request, wrapped_response,
        [call, call_opts, wrapped_request, wrapped_response,
         trace](const Status& status) {
          call->ClearCancelCallback();
          delete call_opts;
          delete wrapped_request;
          delete wrapped_response;
          delete trace;
          if (call->request.store_errors_in_response_body() && !status.ok()) {
            call->response.set_status_code(status.code());
            call->response.set_status_error_message(status.error_message());
            call->SendResponse(ToGrpcStatus(Status::OK()));
          } else {
            call->SendResponse(ToGrpcStatus(status));
          }
        });
    ENQUEUE_REQUEST(RunStep, true);
  }

  // RPC handler for deleting a session.
  void CloseSessionHandler(
      MasterCall<CloseSessionRequest, CloseSessionResponse>* call) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_7(mht_7_v, 409, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "CloseSessionHandler");

    master_impl_->CloseSession(&call->request, &call->response,
                               [call](const Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(CloseSession, false);
  }

  // RPC handler for listing devices.
  void ListDevicesHandler(
      MasterCall<ListDevicesRequest, ListDevicesResponse>* call) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_8(mht_8_v, 422, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "ListDevicesHandler");

    master_impl_->ListDevices(&call->request, &call->response,
                              [call](const Status& status) {
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(ListDevices, false);
  }

  // RPC handler for resetting all sessions.
  void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_9(mht_9_v, 434, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "ResetHandler");

    master_impl_->Reset(&call->request, &call->response,
                        [call](const Status& status) {
                          call->SendResponse(ToGrpcStatus(status));
                        });
    ENQUEUE_REQUEST(Reset, false);
  }

  // RPC handler for making a callable.
  void MakeCallableHandler(
      MasterCall<MakeCallableRequest, MakeCallableResponse>* call) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_10(mht_10_v, 447, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "MakeCallableHandler");

    master_impl_->MakeCallable(&call->request, &call->response,
                               [call](const Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(MakeCallable, false);
  }

  // RPC handler for running a callable.
  void RunCallableHandler(
      MasterCall<RunCallableRequest, RunCallableResponse>* call) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_11(mht_11_v, 460, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "RunCallableHandler");

    auto* trace = TraceRpc("RunCallable/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    // The timeout may be overridden by a non-zero timeout in the
    // callable's `RunOptions`; this overriding will happen inside the
    // `MasterSession` implementation.
    call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunCallable(call_opts, &call->request, &call->response,
                              [call, call_opts, trace](const Status& status) {
                                call->ClearCancelCallback();
                                delete call_opts;
                                delete trace;
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(RunCallable, false);
  }

  // RPC handler for making a callable.
  void ReleaseCallableHandler(
      MasterCall<ReleaseCallableRequest, ReleaseCallableResponse>* call) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_12(mht_12_v, 483, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "ReleaseCallableHandler");

    master_impl_->ReleaseCallable(&call->request, &call->response,
                                  [call](const Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(ReleaseCallable, false);
  }

#undef ENQUEUE_REQUEST

  // Start tracing, including the ID attached to the RPC.
  profiler::TraceMe* TraceRpc(
      StringPiece name,
      const std::multimap<::grpc::string_ref, ::grpc::string_ref>& metadata) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_13(mht_13_v, 499, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "TraceRpc");

    StringPiece id;
    auto it = metadata.find(GrpcIdKey());
    if (it != metadata.end()) {
      id = StringPiece(it->second.data(), it->second.size());
    }
    return new profiler::TraceMe([&] { return strings::StrCat(name, ":", id); },
                                 profiler::TraceMeLevel::kInfo);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterService);
};

AsyncServiceInterface* NewGrpcMasterService(
    Master* master, const ConfigProto& default_session_config,
    ::grpc::ServerBuilder* builder) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_serviceDTcc mht_14(mht_14_v, 517, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service.cc", "NewGrpcMasterService");

  return new GrpcMasterService(master, default_session_config, builder);
}

}  // end namespace tensorflow
