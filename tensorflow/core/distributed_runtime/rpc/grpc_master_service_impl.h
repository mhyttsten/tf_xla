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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh() {
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


#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/completion_queue.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

namespace grpc {

// Implementation of `tensorflow.MasterService`, based on the
// definition in "//tensorflow/core/protobuf/master_service.proto",
// and the gRPC generated stub and service classes.
// See that file for the definition of methods and messages.
class MasterService final {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "~StubInterface");
}
    virtual ::grpc::Status CreateSession(::grpc::ClientContext* context,
                                         const CreateSessionRequest& request,
                                         CreateSessionResponse* response) = 0;
    virtual ::grpc::Status ExtendSession(::grpc::ClientContext* context,
                                         const ExtendSessionRequest& request,
                                         ExtendSessionResponse* response) = 0;
    virtual ::grpc::Status PartialRunSetup(
        ::grpc::ClientContext* context, const PartialRunSetupRequest& request,
        PartialRunSetupResponse* response) = 0;
    virtual ::grpc::Status RunStep(::grpc::ClientContext* context,
                                   const RunStepRequest& request,
                                   RunStepResponse* response) = 0;
    virtual ::grpc::Status CloseSession(::grpc::ClientContext* context,
                                        const CloseSessionRequest& request,
                                        CloseSessionResponse* response) = 0;
    virtual ::grpc::Status ListDevices(::grpc::ClientContext* context,
                                       const ListDevicesRequest& request,
                                       ListDevicesResponse* response) = 0;
    virtual ::grpc::Status Reset(::grpc::ClientContext* context,
                                 const ResetRequest& request,
                                 ResetResponse* response) = 0;
    virtual ::grpc::Status MakeCallable(::grpc::ClientContext* context,
                                        const MakeCallableRequest& request,
                                        MakeCallableResponse* response) = 0;
    virtual ::grpc::Status RunCallable(::grpc::ClientContext* context,
                                       const RunCallableRequest& request,
                                       RunCallableResponse* response) = 0;
    virtual ::grpc::Status ReleaseCallable(
        ::grpc::ClientContext* context, const ReleaseCallableRequest& request,
        ReleaseCallableResponse* response) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status CreateSession(::grpc::ClientContext* context,
                                 const CreateSessionRequest& request,
                                 CreateSessionResponse* response) override;
    ::grpc::Status ExtendSession(::grpc::ClientContext* context,
                                 const ExtendSessionRequest& request,
                                 ExtendSessionResponse* response) override;
    ::grpc::Status PartialRunSetup(::grpc::ClientContext* context,
                                   const PartialRunSetupRequest& request,
                                   PartialRunSetupResponse* response) override;
    ::grpc::Status RunStep(::grpc::ClientContext* context,
                           const RunStepRequest& request,
                           RunStepResponse* response) override;
    ::grpc::Status CloseSession(::grpc::ClientContext* context,
                                const CloseSessionRequest& request,
                                CloseSessionResponse* response) override;
    ::grpc::Status ListDevices(::grpc::ClientContext* context,
                               const ListDevicesRequest& request,
                               ListDevicesResponse* response) override;
    ::grpc::Status Reset(::grpc::ClientContext* context,
                         const ResetRequest& request,
                         ResetResponse* response) override;
    ::grpc::Status MakeCallable(::grpc::ClientContext* context,
                                const MakeCallableRequest& request,
                                MakeCallableResponse* response) override;
    ::grpc::Status RunCallable(::grpc::ClientContext* context,
                               const RunCallableRequest& request,
                               RunCallableResponse* response) override;
    ::grpc::Status ReleaseCallable(::grpc::ClientContext* context,
                                   const ReleaseCallableRequest& request,
                                   ReleaseCallableResponse* response) override;

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::internal::RpcMethod rpcmethod_CreateSession_;
    const ::grpc::internal::RpcMethod rpcmethod_ExtendSession_;
    const ::grpc::internal::RpcMethod rpcmethod_PartialRunSetup_;
    const ::grpc::internal::RpcMethod rpcmethod_RunStep_;
    const ::grpc::internal::RpcMethod rpcmethod_CloseSession_;
    const ::grpc::internal::RpcMethod rpcmethod_ListDevices_;
    const ::grpc::internal::RpcMethod rpcmethod_Reset_;
    const ::grpc::internal::RpcMethod rpcmethod_MakeCallable_;
    const ::grpc::internal::RpcMethod rpcmethod_RunCallable_;
    const ::grpc::internal::RpcMethod rpcmethod_ReleaseCallable_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr< ::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestCreateSession(
        ::grpc::ServerContext* context, CreateSessionRequest* request,
        ::grpc::ServerAsyncResponseWriter<CreateSessionResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_1(mht_1_v, 307, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestCreateSession");

      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestExtendSession(
        ::grpc::ServerContext* context, ExtendSessionRequest* request,
        ::grpc::ServerAsyncResponseWriter<ExtendSessionResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_2(mht_2_v, 318, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestExtendSession");

      ::grpc::Service::RequestAsyncUnary(1, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestPartialRunSetup(
        ::grpc::ServerContext* context, PartialRunSetupRequest* request,
        ::grpc::ServerAsyncResponseWriter<PartialRunSetupResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_3(mht_3_v, 329, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestPartialRunSetup");

      ::grpc::Service::RequestAsyncUnary(2, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRunStep(
        ::grpc::ServerContext* context, RunStepRequest* request,
        ::grpc::ServerAsyncResponseWriter<RunStepResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_4(mht_4_v, 340, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestRunStep");

      ::grpc::Service::RequestAsyncUnary(3, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCloseSession(
        ::grpc::ServerContext* context, CloseSessionRequest* request,
        ::grpc::ServerAsyncResponseWriter<CloseSessionResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_5(mht_5_v, 351, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestCloseSession");

      ::grpc::Service::RequestAsyncUnary(4, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestListDevices(
        ::grpc::ServerContext* context, ListDevicesRequest* request,
        ::grpc::ServerAsyncResponseWriter<ListDevicesResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_6(mht_6_v, 362, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestListDevices");

      ::grpc::Service::RequestAsyncUnary(5, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestReset(
        ::grpc::ServerContext* context, ResetRequest* request,
        ::grpc::ServerAsyncResponseWriter<ResetResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_7(mht_7_v, 373, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestReset");

      ::grpc::Service::RequestAsyncUnary(6, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestMakeCallable(
        ::grpc::ServerContext* context, MakeCallableRequest* request,
        ::grpc::ServerAsyncResponseWriter<MakeCallableResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_8(mht_8_v, 384, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestMakeCallable");

      ::grpc::Service::RequestAsyncUnary(7, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRunCallable(
        ::grpc::ServerContext* context, RunCallableRequest* request,
        ::grpc::ServerAsyncResponseWriter<RunCallableResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_9(mht_9_v, 395, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestRunCallable");

      ::grpc::Service::RequestAsyncUnary(8, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestReleaseCallable(
        ::grpc::ServerContext* context, ReleaseCallableRequest* request,
        ::grpc::ServerAsyncResponseWriter<ReleaseCallableResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTh mht_10(mht_10_v, 406, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h", "RequestReleaseCallable");

      ::grpc::Service::RequestAsyncUnary(9, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_
