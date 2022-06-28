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
// Copied from auto-generated gRPC code in order to enable using grpc_call.h
// for raw message handling.
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_
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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh() {
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


#include <functional>

#include "grpcpp/impl/codegen/async_generic_service.h"
#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/client_callback.h"
#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/completion_queue.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/server_callback.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"

#if defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache.pb.h"
#else
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache.pb.h"  // copybara"
#endif
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"

namespace tensorflow {
namespace tpu {
namespace grpc {
class TpuCompilationCacheService final {
 public:
  using RequestType = ::tensorflow::tpu::GetTpuProgramRequest;
#if defined(LIBTPU_ON_GCE)
  using ResponseType = ::tensorflow::tpu::GetTpuProgramResponseExternal;
#else
  using ResponseType = ::tensorflow::tpu::GetTpuProgramResponse;
#endif

  // N.B. This must be synchronized with the method order in
  // tpu_compilation_cache.proto.
  enum class MethodId { kGetTpuProgram = 0 };

  static constexpr char const* service_full_name() {
#if defined(LIBTPU_ON_GCE)
    return "tensorflow.tpu.TpuCompilationCacheServiceExternal";
#else
    return "tensorflow.tpu.TpuCompilationCacheService";
#endif
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_0(mht_0_v, 239, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "~StubInterface");
}
    // This method requests the cached proto that the TPU execute op has
    // been instructed to execute.
    virtual ::grpc::Status GetTpuProgram(::grpc::ClientContext* context,
                                         const RequestType& request,
                                         ResponseType* response) = 0;
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<ResponseType>>
    AsyncGetTpuProgram(::grpc::ClientContext* context,
                       const RequestType& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReaderInterface<ResponseType>>(
          AsyncGetTpuProgramRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<ResponseType>>
    PrepareAsyncGetTpuProgram(::grpc::ClientContext* context,
                              const RequestType& request,
                              ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReaderInterface<ResponseType>>(
          PrepareAsyncGetTpuProgramRaw(context, request, cq));
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<ResponseType>*
    AsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                          const RequestType& request,
                          ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<ResponseType>*
    PrepareAsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    explicit Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    ::grpc::Status GetTpuProgram(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ResponseType* response) override;
    std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>
    AsyncGetTpuProgram(::grpc::ClientContext* context,
                       const RequestType& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>(
          AsyncGetTpuProgramRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>
    PrepareAsyncGetTpuProgram(::grpc::ClientContext* context,
                              const RequestType& request,
                              ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>(
          PrepareAsyncGetTpuProgramRaw(context, request, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader<ResponseType>* AsyncGetTpuProgramRaw(
        ::grpc::ClientContext* context, const RequestType& request,
        ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader<ResponseType>*
    PrepareAsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_get_tpu_program_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    ~Service() override;
    // This method requests the cached proto that the TPU execute op has
    // been instructed to execute.
    virtual ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                         const RequestType* request,
                                         ResponseType* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_1(mht_1_v, 324, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "BaseClassMustBeDerivedFromService");
}

   public:
    WithAsyncMethod_GetTpuProgram() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_2(mht_2_v, 330, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "WithAsyncMethod_GetTpuProgram");
 ::grpc::Service::MarkMethodAsync(0); }
    ~WithAsyncMethod_GetTpuProgram() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_3(mht_3_v, 334, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "~WithAsyncMethod_GetTpuProgram");

      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_4(mht_4_v, 343, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "GetTpuProgram");

      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetTpuProgram(
        ::grpc::ServerContext* context, RequestType* request,
        ::grpc::ServerAsyncResponseWriter<ResponseType>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_5(mht_5_v, 354, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "RequestGetTpuProgram");

      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }

    // Make RequestAsyncUnary accessible to grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
  typedef WithAsyncMethod_GetTpuProgram<Service> AsyncService;
  template <class BaseClass>
  class WithGenericMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_6(mht_6_v, 369, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "BaseClassMustBeDerivedFromService");
}

   public:
    WithGenericMethod_GetTpuProgram() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_7(mht_7_v, 375, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "WithGenericMethod_GetTpuProgram");
 ::grpc::Service::MarkMethodGeneric(0); }
    ~WithGenericMethod_GetTpuProgram() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_8(mht_8_v, 379, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "~WithGenericMethod_GetTpuProgram");

      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_9(mht_9_v, 388, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "GetTpuProgram");

      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_10(mht_10_v, 399, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "BaseClassMustBeDerivedFromService");
}

   public:
    WithStreamedUnaryMethod_GetTpuProgram() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_11(mht_11_v, 405, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "WithStreamedUnaryMethod_GetTpuProgram");

      ::grpc::Service::MarkMethodStreamed(
          0,
          new ::grpc::internal::StreamedUnaryHandler<RequestType, ResponseType>(
              std::bind(&WithStreamedUnaryMethod_GetTpuProgram<
                            BaseClass>::StreamedGetTpuProgram,
                        this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetTpuProgram() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_12(mht_12_v, 416, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "~WithStreamedUnaryMethod_GetTpuProgram");

      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_grpcDTh mht_13(mht_13_v, 425, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h", "GetTpuProgram");

      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetTpuProgram(
        ::grpc::ServerContext* context,
        ::grpc::ServerUnaryStreamer<RequestType, ResponseType>*
            server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_GetTpuProgram<Service> StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_GetTpuProgram<Service> StreamedService;
};
}  // namespace grpc
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_
