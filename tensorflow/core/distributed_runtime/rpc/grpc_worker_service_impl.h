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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_service_implDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_service_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_service_implDTh() {
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
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/support/byte_buffer.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace grpc {

// Support parsing/unparsing of tensorflow::TensorResponse.
// Wire-format is identical to RecvTensorResponse.
// This is specializing an existing template, so it's okay to do this in a
// namespace that we don't own.
template <>
class SerializationTraits<tensorflow::TensorResponse> {
 public:
  static Status Serialize(const tensorflow::TensorResponse& msg, ByteBuffer* bp,
                          bool* own_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_service_implDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h", "Serialize");

    LOG(FATAL) << "TODO(sanjay,jeff): Implement";
    return Status();
  }
  static Status Deserialize(ByteBuffer* buffer,
                            tensorflow::TensorResponse* msg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_service_implDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h", "Deserialize");

    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }
    Status result = Status::OK;
    if (result.ok()) {
      ::tensorflow::GrpcByteSource source(buffer);
      auto s = msg->ParseFrom(&source);
      if (!s.ok()) {
        result = Status(StatusCode::INTERNAL,
                        ::tensorflow::strings::StrCat(
                            "TensorResponse parse error", s.error_message()));
      }
    }
    buffer->Clear();
    return result;
  }
};

}  // namespace grpc

namespace tensorflow {

// Names of worker methods.
enum class GrpcWorkerMethod {
  kGetStatus,
  kCreateWorkerSession,
  kDeleteWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kRecvBuf,
  kLogging,
  kTracing,
  kCompleteGroup,
  kCompleteInstance,
  kGetStepSequence,
  kMarkRecvFinished,
};

static const int kGrpcNumWorkerMethods =
    static_cast<int>(GrpcWorkerMethod::kMarkRecvFinished) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc {

// Implementation of `tensorflow.WorkerService`, based on the
// definition in "//tensorflow/core/protobuf/worker_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class WorkerService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
