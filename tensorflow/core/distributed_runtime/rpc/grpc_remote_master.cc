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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"

#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// GrpcRemoteMaster is an implementation of the MasterInterface
// that uses gRPC to talk to the Master service.
class GrpcRemoteMaster : public MasterInterface {
  using MasterServiceStub = grpc::MasterService::Stub;

 public:
  explicit GrpcRemoteMaster(const SharedGrpcChannelPtr& client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "GrpcRemoteMaster");
}

  ~GrpcRemoteMaster() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "~GrpcRemoteMaster");
}

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "CreateSession");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CreateSession);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "ExtendSession");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ExtendSession);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "PartialRunSetup");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::PartialRunSetup);
  }

  Status RunStep(CallOptions* call_options, RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "RunStep");

    return CallWithRetry(call_options, &request->ToProto(),
                         get_proto_from_wrapper(response),
                         &MasterServiceStub::RunStep, "RunStep/Client");
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_6(mht_6_v, 264, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "CloseSession");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CloseSession);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_7(mht_7_v, 274, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "ListDevices");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ListDevices);
  }

  Status Reset(CallOptions* call_options, const ResetRequest* request,
               ResetResponse* response) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_8(mht_8_v, 283, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "Reset");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::Reset);
  }

  Status MakeCallable(CallOptions* call_options,
                      const MakeCallableRequest* request,
                      MakeCallableResponse* response) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_9(mht_9_v, 293, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "MakeCallable");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::MakeCallable);
  }
  Status RunCallable(CallOptions* call_options,
                     const RunCallableRequest* request,
                     RunCallableResponse* response) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_10(mht_10_v, 302, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "RunCallable");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::RunCallable);
  }
  Status ReleaseCallable(CallOptions* call_options,
                         const ReleaseCallableRequest* request,
                         ReleaseCallableResponse* response) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_11(mht_11_v, 311, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "ReleaseCallable");

    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ReleaseCallable);
  }

 private:
  // Start tracing, attaching a unique ID to both the trace and the RPC.
  profiler::TraceMe* NewTraceRpc(StringPiece name, ::grpc::ClientContext* ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_12(mht_12_v, 321, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "NewTraceRpc");

    string trace_id = strings::StrCat(tracing::GetUniqueArg());
    ctx->AddMetadata(GrpcIdKey(), trace_id);
    return new profiler::TraceMe(
        [&] { return strings::StrCat(name, ":", trace_id); },
        profiler::TraceMeLevel::kInfo);
  }

  template <typename Request, typename Response>
  Status CallWithRetry(CallOptions* call_options, const Request* request,
                       Response* response,
                       ::grpc::Status (MasterServiceStub::*pfunc)(
                           ::grpc::ClientContext*, const Request&, Response*),
                       string trace_string = {}) {
    absl::Duration timeout = absl::Milliseconds(call_options->GetTimeout());
    absl::Time expired_time = absl::FromUnixMicros(Env::Default()->NowMicros());
    if (timeout > absl::ZeroDuration()) {
      expired_time += timeout;
    }
    Status s;
    for (int num_retries = 0;; ++num_retries) {
      ::grpc::ClientContext ctx;
      std::unique_ptr<profiler::TraceMe> trace;
      if (!trace_string.empty()) {
        trace.reset(NewTraceRpc(trace_string, &ctx));
      }
      ctx.set_fail_fast(false);
      if (timeout > absl::ZeroDuration()) {
        // We do not modify the timeout here to match legacy behavior. However,
        // this could violate the contract of tensorflow::Session. If we retry
        // an RPC just before the deadline is exceeded, we will still set the
        // timeout to the original value. This leads to the overall timeout
        // being double what was expected.
        ctx.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
      }
      s = FromGrpcStatus((stub_.get()->*pfunc)(&ctx, *request, response));
      if (!errors::IsUnavailable(s)) {
        return s;
      }
      // TODO(b/117162170): we may want to make this configurable.
      constexpr int kMaxRetries = 10;
      LOG(WARNING) << "RPC failed with status = \"" << s
                   << "\" and grpc_error_string = \""
                   << ctx.debug_error_string() << "\", maybe retrying the RPC";
      if (num_retries >= kMaxRetries) {
        LOG(WARNING) << "Too many retries, returning last status: " << s;
        return s;
      }
      absl::Time now = absl::FromUnixMicros(Env::Default()->NowMicros());
      const absl::Time deadline_with_backoff =
          now + absl::Microseconds(ComputeBackoffMicroseconds(num_retries));
      // Wait for a short period of time before retrying the RPC.  If our
      // backoff would put us past the RPC deadline, we truncate it to ensure
      // our RPC starts before the deadline.
      const auto backoff_until = (timeout <= absl::ZeroDuration() ||
                                  expired_time > deadline_with_backoff)
                                     ? deadline_with_backoff
                                     : expired_time;
      Env::Default()->SleepForMicroseconds(
          absl::ToInt64Microseconds(backoff_until - now));
      now = absl::FromUnixMicros(Env::Default()->NowMicros());
      if (now > expired_time && timeout > absl::ZeroDuration()) {
        // If timeout_in_ms is set, exit the retry loop on timeout.
        return errors::DeadlineExceeded(ctx.debug_error_string());
      }
    }
  }

  std::unique_ptr<MasterServiceStub> stub_;
};

MasterInterface* NewGrpcMaster(const SharedGrpcChannelPtr& channel) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_remote_masterDTcc mht_13(mht_13_v, 395, "", "./tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc", "NewGrpcMaster");

  return new GrpcRemoteMaster(channel);
}

}  // namespace tensorflow
