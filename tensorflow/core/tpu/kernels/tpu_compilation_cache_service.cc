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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_service.h"

#include <chrono>  // NOLINT

#include "grpcpp/support/byte_buffer.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"

namespace tensorflow {
namespace {
using ::tensorflow::tpu::CompilationCacheEntryRef;
using ::tensorflow::tpu::TpuCompilationCacheEntry;
using ::tensorflow::tpu::TpuCompilationCacheInterface;

static constexpr int kGetTpuProgramServingThreads = 32;
}  // namespace

TpuCompilationCacheService::TpuCompilationCacheService(
    ::grpc::ServerBuilder* server_builder, TpuCompilationCacheInterface* cache)
    : running_(true),
      cache_(cache),
      server_builder_(server_builder),
      cq_(server_builder_->AddCompletionQueue()),
      thread_pool_(absl::make_unique<thread::ThreadPool>(
          Env::Default(), "TpuCompilationCacheService",
          kGetTpuProgramServingThreads)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::TpuCompilationCacheService");

  cache_->Ref();
  server_builder_->RegisterService(&service_);
}

TpuCompilationCacheService::~TpuCompilationCacheService() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::~TpuCompilationCacheService");

  // This ordering is important. We must first shutdown our CQ and allow the
  // polling thread and dispatch pool to shutdown before releasing our cache
  // reference. The gRPC server must be Shutdown() by this point or we will
  // deadlock here.  The running_ boolean is necessary to avoid adding new
  // operations to the CQ after is has shutdown.
  running_ = false;
  cq_->Shutdown();
  polling_thread_.reset();
  thread_pool_.reset();
  cache_->Unref();
}

void TpuCompilationCacheService::Start() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::Start");

  server_ = server_builder_->BuildAndStart();
  ThreadOptions opts;
  polling_thread_.reset(Env::Default()->StartThread(
      opts, "TpuCompilationCachePoller", [this]() { HandleRPCsLoop(); }));
}

bool TpuCompilationCacheService::Shutdown(int timeout_sec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::Shutdown");

  if (server_ != nullptr) {
    std::chrono::system_clock::time_point timeout =
        std::chrono::system_clock::now() + std::chrono::seconds(timeout_sec);
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(timeout_sec));
    if (std::chrono::system_clock::now() >= timeout) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

void TpuCompilationCacheService::SetMemoryQuota(size_t max_bytes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::SetMemoryQuota");

  ::grpc::ResourceQuota quota;
  quota.Resize(max_bytes);
  server_builder_->SetResourceQuota(quota);
}

// Fetch a cache result for the given request and serialize the result directly
// into a ByteBuffer.
void TpuCompilationCacheService::GetTpuProgram(GetTpuProgramCall* call) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_5(mht_5_v, 273, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::GetTpuProgram");

  std::unique_ptr<CompilationCacheEntryRef> entry;

  VLOG(1) << "GetTpuProgram: " << call->request.DebugString();
  Status s;
  switch (call->request.key_oneof_case()) {
    case tpu::GetTpuProgramRequest::kKey:
      s = cache_->Lookup(call->request.key(), &entry);
      break;

    case tpu::GetTpuProgramRequest::kUidAndIndex:
      s = cache_->Lookup(call->request.uid_and_index().uid(),
                         call->request.uid_and_index().proto_index(), &entry);
      break;

    default:
      s = errors::Internal("Bad GetTpuProgram RPC request oneof case ",
                           call->request.key_oneof_case());
      break;
  }
  if (!s.ok()) {
    return call->SendResponse(ToGrpcStatus(s));
  }

  s = entry->ToSubEntryRef(call->request.fetch_target());
  if (!s.ok()) {
    return call->SendResponse(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat(
            "Error getting the fetching target ",
            CompilationCacheFetchTarget_Name(call->request.fetch_target())),
        s.error_message()));
  }

  TpuCompilationCacheEntry cache_entry = entry->get();
  if (cache_entry.tpu_program_group() == nullptr) {
    // It's possible that the sharding/unsharding entry does not exist, but the
    // main entry must exist.
    CHECK_NE(call->request.fetch_target(),
             tpu::CompilationCacheFetchTarget::MAIN);
  }

  xla::StatusOr<std::vector<::grpc::Slice>> buffer_slices =
      tpu::SerializeCacheEntryToBufferSlices(cache_entry);

  if (!buffer_slices.ok()) {
    return call->SendResponse(ToGrpcStatus(buffer_slices.status()));
  }

  call->response =
      ::grpc::ByteBuffer{&buffer_slices.ValueOrDie()[0], buffer_slices->size()};
  return call->SendResponse(::grpc::Status());
}

void TpuCompilationCacheService::HandleGetTpuProgram(GetTpuProgramCall* call) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_6(mht_6_v, 330, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::HandleGetTpuProgram");

  thread_pool_->Schedule([this, call]() { GetTpuProgram(call); });
  if (running_) {
    GetTpuProgramCall::EnqueueRequestForMethod(
        &service_, cq_.get(),
        static_cast<int>(ServiceType::MethodId::kGetTpuProgram),
        &TpuCompilationCacheService::HandleGetTpuProgram,
        /*supports_cancel=*/false);
  }
}

void TpuCompilationCacheService::HandleRPCsLoop() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_serviceDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_service.cc", "TpuCompilationCacheService::HandleRPCsLoop");

  void* tag;
  bool ok;

  for (int i = 0; i < 50; ++i) {
    GetTpuProgramCall::EnqueueRequestForMethod(
        &service_, cq_.get(),
        static_cast<int>(ServiceType::MethodId::kGetTpuProgram),
        &TpuCompilationCacheService::HandleGetTpuProgram,
        /*supports_cancel=*/false);
  }

  while (cq_->Next(&tag, &ok)) {
    VLOG(2) << "HandleRPCS: " << tag;
    UntypedCall<TpuCompilationCacheService>::Tag* callback_tag =
        static_cast<UntypedCall<TpuCompilationCacheService>::Tag*>(tag);
    callback_tag->OnCompleted(this, ok);
  }

  VLOG(2) << "Cache thread shutting down.";
}
}  // namespace tensorflow
