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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {

namespace grpc {

static const char* grpcMasterService_method_names[] = {
    "/tensorflow.MasterService/CreateSession",
    "/tensorflow.MasterService/ExtendSession",
    "/tensorflow.MasterService/PartialRunSetup",
    "/tensorflow.MasterService/RunStep",
    "/tensorflow.MasterService/CloseSession",
    "/tensorflow.MasterService/ListDevices",
    "/tensorflow.MasterService/Reset",
    "/tensorflow.MasterService/MakeCallable",
    "/tensorflow.MasterService/RunCallable",
    "/tensorflow.MasterService/ReleaseCallable",
};

std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::NewStub");

  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

MasterService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_CreateSession_(grpcMasterService_method_names[0],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_ExtendSession_(grpcMasterService_method_names[1],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_PartialRunSetup_(grpcMasterService_method_names[2],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel),
      rpcmethod_RunStep_(grpcMasterService_method_names[3],
                         ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CloseSession_(grpcMasterService_method_names[4],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ListDevices_(grpcMasterService_method_names[5],
                             ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Reset_(grpcMasterService_method_names[6],
                       ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_MakeCallable_(grpcMasterService_method_names[7],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RunCallable_(grpcMasterService_method_names[8],
                             ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ReleaseCallable_(grpcMasterService_method_names[9],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::Stub");
}

::grpc::Status MasterService::Stub::CreateSession(
    ::grpc::ClientContext* context, const CreateSessionRequest& request,
    CreateSessionResponse* response) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::CreateSession");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CreateSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ExtendSession(
    ::grpc::ClientContext* context, const ExtendSessionRequest& request,
    ExtendSessionResponse* response) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::ExtendSession");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ExtendSession_, context, request, response);
}

::grpc::Status MasterService::Stub::PartialRunSetup(
    ::grpc::ClientContext* context, const PartialRunSetupRequest& request,
    PartialRunSetupResponse* response) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::PartialRunSetup");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_PartialRunSetup_, context, request, response);
}

::grpc::Status MasterService::Stub::RunStep(::grpc::ClientContext* context,
                                            const RunStepRequest& request,
                                            RunStepResponse* response) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::RunStep");

  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_RunStep_,
                                             context, request, response);
}

::grpc::Status MasterService::Stub::CloseSession(
    ::grpc::ClientContext* context, const CloseSessionRequest& request,
    CloseSessionResponse* response) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::CloseSession");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CloseSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ListDevices(
    ::grpc::ClientContext* context, const ListDevicesRequest& request,
    ListDevicesResponse* response) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_7(mht_7_v, 306, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::ListDevices");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ListDevices_, context, request, response);
}

::grpc::Status MasterService::Stub::Reset(::grpc::ClientContext* context,
                                          const ResetRequest& request,
                                          ResetResponse* response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::Reset");

  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_Reset_,
                                             context, request, response);
}

::grpc::Status MasterService::Stub::MakeCallable(
    ::grpc::ClientContext* context, const MakeCallableRequest& request,
    MakeCallableResponse* response) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_9(mht_9_v, 326, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::MakeCallable");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_MakeCallable_, context, request, response);
}

::grpc::Status MasterService::Stub::RunCallable(
    ::grpc::ClientContext* context, const RunCallableRequest& request,
    RunCallableResponse* response) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_10(mht_10_v, 336, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::RunCallable");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_RunCallable_, context, request, response);
}

::grpc::Status MasterService::Stub::ReleaseCallable(
    ::grpc::ClientContext* context, const ReleaseCallableRequest& request,
    ReleaseCallableResponse* response) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_11(mht_11_v, 346, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::Stub::ReleaseCallable");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ReleaseCallable_, context, request, response);
}

MasterService::AsyncService::AsyncService() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_12(mht_12_v, 354, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::AsyncService::AsyncService");

  int method_len = sizeof(grpcMasterService_method_names) / 
                    sizeof(grpcMasterService_method_names[0]);
  for (int i = 0; i < method_len; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcMasterService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_master_service_implDTcc mht_13(mht_13_v, 368, "", "./tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc", "MasterService::AsyncService::~AsyncService");
}

}  // namespace grpc

}  // namespace tensorflow
