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
class MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc() {
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

#include "tensorflow/compiler/xla/rpc/grpc_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<GRPCService>> GRPCService::NewService(
    se::Platform* platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_0(mht_0_v, 192, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::NewService");

  std::unique_ptr<GRPCService> grpc_service(new GRPCService());
  TF_ASSIGN_OR_RETURN(grpc_service->service_,
                      ::xla::Service::NewService(platform));
  return std::move(grpc_service);
}

::grpc::Status DelegateRPC(std::function<Status()> op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_1(mht_1_v, 202, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "DelegateRPC");

  Status s = op();
  return tensorflow::ToGrpcStatus(s);
}

::grpc::Status GRPCService::Unregister(::grpc::ServerContext* context,
                                       const UnregisterRequest* arg,
                                       UnregisterResponse* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_2(mht_2_v, 212, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::Unregister");

  return DelegateRPC(
      [this, arg, result]() { return service_->Unregister(arg, result); });
}

::grpc::Status GRPCService::DeconstructTuple(::grpc::ServerContext* context,
                                             const DeconstructTupleRequest* arg,
                                             DeconstructTupleResponse* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_3(mht_3_v, 222, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::DeconstructTuple");

  return DelegateRPC([this, arg, result]() {
    return service_->DeconstructTuple(arg, result);
  });
}

::grpc::Status GRPCService::GetDeviceHandles(::grpc::ServerContext* context,
                                             const GetDeviceHandlesRequest* arg,
                                             GetDeviceHandlesResponse* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_4(mht_4_v, 233, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::GetDeviceHandles");

  return DelegateRPC([this, arg, result]() {
    return service_->GetDeviceHandles(arg, result);
  });
}

::grpc::Status GRPCService::Compile(::grpc::ServerContext* /*context*/,
                                    const CompileRequest* arg,
                                    CompileResponse* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_5(mht_5_v, 244, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::Compile");

  return DelegateRPC(
      [this, arg, result]() { return service_->Compile(arg, result); });
}

::grpc::Status GRPCService::Execute(::grpc::ServerContext* /*context*/,
                                    const ExecuteRequest* arg,
                                    ExecuteResponse* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_6(mht_6_v, 254, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::Execute");

  return DelegateRPC(
      [this, arg, result]() { return service_->Execute(arg, result); });
}

::grpc::Status GRPCService::ExecuteGraphParallel(
    ::grpc::ServerContext* /*context*/, const ExecuteGraphParallelRequest* arg,
    ExecuteParallelResponse* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_7(mht_7_v, 264, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::ExecuteGraphParallel");

  return DelegateRPC([this, arg, result]() {
    return service_->ExecuteGraphParallel(arg, result);
  });
}

::grpc::Status GRPCService::WaitForExecution(::grpc::ServerContext* context,
                                             const WaitForExecutionRequest* arg,
                                             WaitForExecutionResponse* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_8(mht_8_v, 275, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::WaitForExecution");

  return DelegateRPC([this, arg, result]() {
    return service_->WaitForExecution(arg, result);
  });
}

::grpc::Status GRPCService::TransferToClient(::grpc::ServerContext* context,
                                             const TransferToClientRequest* arg,
                                             TransferToClientResponse* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_9(mht_9_v, 286, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::TransferToClient");

  return DelegateRPC([this, arg, result]() {
    return service_->TransferToClient(arg, result);
  });
}

::grpc::Status GRPCService::TransferToServer(::grpc::ServerContext* context,
                                             const TransferToServerRequest* arg,
                                             TransferToServerResponse* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_10(mht_10_v, 297, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::TransferToServer");

  return DelegateRPC([this, arg, result]() {
    return service_->TransferToServer(arg, result);
  });
}

::grpc::Status GRPCService::TransferToInfeed(::grpc::ServerContext* context,
                                             const TransferToInfeedRequest* arg,
                                             TransferToInfeedResponse* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_11(mht_11_v, 308, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::TransferToInfeed");

  return DelegateRPC([this, arg, result]() {
    return service_->TransferToInfeed(arg, result);
  });
}

::grpc::Status GRPCService::TransferFromOutfeed(
    ::grpc::ServerContext* context, const TransferFromOutfeedRequest* arg,
    TransferFromOutfeedResponse* result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_12(mht_12_v, 319, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::TransferFromOutfeed");

  return DelegateRPC([this, arg, result]() {
    return service_->TransferFromOutfeed(arg, result);
  });
}

::grpc::Status GRPCService::ResetDevice(::grpc::ServerContext* context,
                                        const ResetDeviceRequest* arg,
                                        ResetDeviceResponse* result) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_13(mht_13_v, 330, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::ResetDevice");

  return DelegateRPC(
      [this, arg, result]() { return service_->ResetDevice(arg, result); });
}

::grpc::Status GRPCService::GetShape(::grpc::ServerContext* context,
                                     const GetShapeRequest* arg,
                                     GetShapeResponse* result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_serviceDTcc mht_14(mht_14_v, 340, "", "./tensorflow/compiler/xla/rpc/grpc_service.cc", "GRPCService::GetShape");

  return DelegateRPC(
      [this, arg, result]() { return service_->GetShape(arg, result); });
}

}  // namespace xla
