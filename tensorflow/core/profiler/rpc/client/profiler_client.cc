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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc() {
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

/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"

#include <limits>

#include "grpcpp/grpcpp.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

inline Status FromGrpcStatus(const ::grpc::Status& s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "FromGrpcStatus");

  return s.ok() ? Status::OK()
                : Status(static_cast<error::Code>(s.error_code()),
                         s.error_message());
}

template <typename T>
std::unique_ptr<typename T::Stub> CreateStub(
    const std::string& service_address) {
  ::grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  // Default URI prefix is "dns:///" if not provided.
  auto channel = ::grpc::CreateCustomChannel(
      service_address, ::grpc::InsecureChannelCredentials(), channel_args);
  if (!channel) {
    LOG(ERROR) << "Unable to create channel" << service_address;
  }
  return T::NewStub(channel);
}

}  // namespace

Status ProfileGrpc(const std::string& service_address,
                   const ProfileRequest& request, ProfileResponse* response) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("service_address: \"" + service_address + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "ProfileGrpc");

  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      CreateStub<grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Profile(&context, request, response)));
  return Status::OK();
}

Status NewSessionGrpc(const std::string& service_address,
                      const NewProfileSessionRequest& request,
                      NewProfileSessionResponse* response) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("service_address: \"" + service_address + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "NewSessionGrpc");

  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfileAnalysis::Stub> stub =
      CreateStub<grpc::ProfileAnalysis>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->NewSession(&context, request, response)));
  return Status::OK();
}

Status MonitorGrpc(const std::string& service_address,
                   const MonitorRequest& request, MonitorResponse* response) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("service_address: \"" + service_address + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "MonitorGrpc");

  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      CreateStub<grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Monitor(&context, request, response)));
  return Status::OK();
}

/*static*/ std::unique_ptr<RemoteProfilerSession> RemoteProfilerSession::Create(
    const std::string& service_address, absl::Time deadline,
    const ProfileRequest& profile_request) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("service_address: \"" + service_address + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "RemoteProfilerSession::Create");

  auto instance = absl::WrapUnique(
      new RemoteProfilerSession(service_address, deadline, profile_request));
  instance->ProfileAsync();
  return instance;
}

RemoteProfilerSession::RemoteProfilerSession(
    const std::string& service_address, absl::Time deadline,
    const ProfileRequest& profile_request)
    : response_(absl::make_unique<ProfileResponse>()),
      service_address_(service_address),
      stub_(CreateStub<grpc::ProfilerService>(service_address_)),
      deadline_(deadline),
      profile_request_(profile_request) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("service_address: \"" + service_address + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "RemoteProfilerSession::RemoteProfilerSession");

  response_->set_empty_trace(true);
}

RemoteProfilerSession::~RemoteProfilerSession() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "RemoteProfilerSession::~RemoteProfilerSession");

  Status dummy;
  WaitForCompletion(dummy);
  grpc_context_.TryCancel();
}

void RemoteProfilerSession::ProfileAsync() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "RemoteProfilerSession::ProfileAsync");

  LOG(INFO) << "Asynchronous gRPC Profile() to " << service_address_;
  grpc_context_.set_deadline(absl::ToChronoTime(deadline_));
  VLOG(1) << "Deadline set to " << deadline_;
  rpc_ = stub_->AsyncProfile(&grpc_context_, profile_request_, &cq_);
  // Connection failure will create lame channel whereby grpc_status_ will be an
  // error.
  rpc_->Finish(response_.get(), &grpc_status_,
               static_cast<void*>(&status_on_completion_));
  VLOG(2) << "Asynchronous gRPC Profile() issued." << absl::Now();
}

std::unique_ptr<ProfileResponse> RemoteProfilerSession::WaitForCompletion(
    Status& out_status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTcc mht_8(mht_8_v, 323, "", "./tensorflow/core/profiler/rpc/client/profiler_client.cc", "RemoteProfilerSession::WaitForCompletion");

  if (!response_) {
    out_status = errors::FailedPrecondition(
        "WaitForCompletion must only be called once.");
    return nullptr;
  }
  LOG(INFO) << "Waiting for completion.";

  void* got_tag = nullptr;
  bool ok = false;
  // Next blocks until there is a response in the completion queue. Expect the
  // completion queue to have exactly a single response because deadline is set
  // and completion queue is only drained once at destruction time.
  bool success = cq_.Next(&got_tag, &ok);
  if (!success || !ok || got_tag == nullptr) {
    out_status =
        errors::Internal("Missing or invalid event from completion queue.");
    return nullptr;
  }

  VLOG(1) << "Writing out status.";
  // For the event read from the completion queue, expect that got_tag points to
  // the memory location of status_on_completion.
  DCHECK_EQ(got_tag, &status_on_completion_);
  // tagged status points to pre-allocated memory which is okay to overwrite.
  status_on_completion_.Update(FromGrpcStatus(grpc_status_));
  if (status_on_completion_.code() == error::DEADLINE_EXCEEDED) {
    LOG(WARNING) << status_on_completion_;
  } else if (!status_on_completion_.ok()) {
    LOG(ERROR) << status_on_completion_;
  }

  out_status = status_on_completion_;
  return std::move(response_);
}

}  // namespace profiler
}  // namespace tensorflow
