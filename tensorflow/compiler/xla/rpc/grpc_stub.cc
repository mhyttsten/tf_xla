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
class MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc() {
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

#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace xla {

GRPCStub::~GRPCStub() = default;

Status MakeRPC(
    const std::function<::grpc::Status(::grpc::ClientContext*)>& rpc_method) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "MakeRPC");

  ::grpc::ClientContext context;
  ::grpc::Status s = rpc_method(&context);
  return tensorflow::FromGrpcStatus(s);
}

Status GRPCStub::TransferToClient(const TransferToClientRequest* request,
                                  TransferToClientResponse* response) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_1(mht_1_v, 203, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::TransferToClient");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToClient(context, *request, response);
  });
}

Status GRPCStub::TransferToServer(const TransferToServerRequest* request,
                                  TransferToServerResponse* response) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_2(mht_2_v, 213, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::TransferToServer");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToServer(context, *request, response);
  });
}

Status GRPCStub::TransferToInfeed(const TransferToInfeedRequest* request,
                                  TransferToInfeedResponse* response) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_3(mht_3_v, 223, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::TransferToInfeed");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToInfeed(context, *request, response);
  });
}

Status GRPCStub::TransferFromOutfeed(const TransferFromOutfeedRequest* request,
                                     TransferFromOutfeedResponse* response) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_4(mht_4_v, 233, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::TransferFromOutfeed");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferFromOutfeed(context, *request, response);
  });
}

Status GRPCStub::ResetDevice(const ResetDeviceRequest* request,
                             ResetDeviceResponse* response) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_5(mht_5_v, 243, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::ResetDevice");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ResetDevice(context, *request, response);
  });
}

Status GRPCStub::Compile(const CompileRequest* request,
                         CompileResponse* response) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_6(mht_6_v, 253, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::Compile");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Compile(context, *request, response);
  });
}

Status GRPCStub::Execute(const ExecuteRequest* request,
                         ExecuteResponse* response) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_7(mht_7_v, 263, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::Execute");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Execute(context, *request, response);
  });
}

Status GRPCStub::ExecuteGraphParallel(
    const ExecuteGraphParallelRequest* request,
    ExecuteParallelResponse* response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_8(mht_8_v, 274, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::ExecuteGraphParallel");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteGraphParallel(context, *request, response);
  });
}

Status GRPCStub::WaitForExecution(const WaitForExecutionRequest* request,
                                  WaitForExecutionResponse* response) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_9(mht_9_v, 284, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::WaitForExecution");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->WaitForExecution(context, *request, response);
  });
}

Status GRPCStub::DeconstructTuple(const DeconstructTupleRequest* request,
                                  DeconstructTupleResponse* response) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_10(mht_10_v, 294, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::DeconstructTuple");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DeconstructTuple(context, *request, response);
  });
}

Status GRPCStub::GetComputationGraphStats(
    const ComputationGraphStatsRequest* request,
    ComputationStatsResponse* response) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_11(mht_11_v, 305, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::GetComputationGraphStats");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetComputationGraphStats(context, *request, response);
  });
}

Status GRPCStub::GetShape(const GetShapeRequest* request,
                          GetShapeResponse* response) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_12(mht_12_v, 315, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::GetShape");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetShape(context, *request, response);
  });
}

Status GRPCStub::GetDeviceHandles(const GetDeviceHandlesRequest* request,
                                  GetDeviceHandlesResponse* response) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_13(mht_13_v, 325, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::GetDeviceHandles");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetDeviceHandles(context, *request, response);
  });
}

Status GRPCStub::CreateChannelHandle(const CreateChannelHandleRequest* request,
                                     CreateChannelHandleResponse* response) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_14(mht_14_v, 335, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::CreateChannelHandle");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->CreateChannelHandle(context, *request, response);
  });
}

Status GRPCStub::ComputeConstantGraph(
    const ComputeConstantGraphRequest* request,
    ComputeConstantResponse* response) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_15(mht_15_v, 346, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::ComputeConstantGraph");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ComputeConstantGraph(context, *request, response);
  });
}

// Methods used by GlobalData.
Status GRPCStub::Unregister(const UnregisterRequest* request,
                            UnregisterResponse* response) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSrpcPSgrpc_stubDTcc mht_16(mht_16_v, 357, "", "./tensorflow/compiler/xla/rpc/grpc_stub.cc", "GRPCStub::Unregister");

  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Unregister(context, *request, response);
  });
}

}  // namespace xla
