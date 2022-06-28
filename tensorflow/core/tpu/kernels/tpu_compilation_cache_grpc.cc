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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h"

#include <functional>

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_callback.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/server_callback.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {
namespace tpu {

static const char* grpcTpuCompilationCacheService_method_names[] = {
#if defined(LIBTPU_ON_GCE)
    "/tensorflow.tpu.TpuCompilationCacheServiceExternal/GetTpuProgram",
#else  // LIBTPU_ON_GCE
    "/tensorflow.tpu.TpuCompilationCacheService/GetTpuProgram",
#endif  // LIBTPU_ON_GCE
};

std::unique_ptr<grpc::TpuCompilationCacheService::Stub>
grpc::TpuCompilationCacheService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::NewStub");

  (void)options;
  std::unique_ptr<grpc::TpuCompilationCacheService::Stub> stub(
      new grpc::TpuCompilationCacheService::Stub(channel));
  return stub;
}

grpc::TpuCompilationCacheService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_get_tpu_program_(grpcTpuCompilationCacheService_method_names[0],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Stub::Stub");
}

::grpc::Status grpc::TpuCompilationCacheService::Stub::GetTpuProgram(
    ::grpc::ClientContext* context, const RequestType& request,
    ResponseType* response) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Stub::GetTpuProgram");

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_get_tpu_program_, context, request, response);
}

::grpc::ClientAsyncResponseReader<
    grpc::TpuCompilationCacheService::ResponseType>*
grpc::TpuCompilationCacheService::Stub::AsyncGetTpuProgramRaw(
    ::grpc::ClientContext* context, const RequestType& request,
    ::grpc::CompletionQueue* cq) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Stub::AsyncGetTpuProgramRaw");

  return ::grpc::internal::ClientAsyncResponseReaderFactory<
      ResponseType>::Create(channel_.get(), cq, rpcmethod_get_tpu_program_,
                            context, request, true);
}

::grpc::ClientAsyncResponseReader<
    grpc::TpuCompilationCacheService::ResponseType>*
grpc::TpuCompilationCacheService::Stub::PrepareAsyncGetTpuProgramRaw(
    ::grpc::ClientContext* context, const RequestType& request,
    ::grpc::CompletionQueue* cq) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Stub::PrepareAsyncGetTpuProgramRaw");

  return ::grpc::internal::ClientAsyncResponseReaderFactory<
      ResponseType>::Create(channel_.get(), cq, rpcmethod_get_tpu_program_,
                            context, request, false);
}

grpc::TpuCompilationCacheService::Service::Service() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Service::Service");

  AddMethod(new ::grpc::internal::RpcServiceMethod(
      grpcTpuCompilationCacheService_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler<
          grpc::TpuCompilationCacheService::Service, RequestType, ResponseType>(
          std::mem_fn(
              &grpc::TpuCompilationCacheService::Service::GetTpuProgram),
          this)));
}

grpc::TpuCompilationCacheService::Service::~Service() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Service::~Service");
}

::grpc::Status grpc::TpuCompilationCacheService::Service::GetTpuProgram(
    ::grpc::ServerContext* context, const RequestType* request,
    ResponseType* response) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTcc mht_7(mht_7_v, 290, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.cc", "grpc::TpuCompilationCacheService::Service::GetTpuProgram");

  (void)context;
  (void)request;
  (void)response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

}  // namespace tpu
}  // namespace tensorflow
