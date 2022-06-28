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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc() {
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

#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"

#include <memory>

#include "grpcpp/support/status.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/file_system_utils.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kXPlanePb = "xplane.pb";

// Collects data in XSpace format. The data is saved to a repository
// unconditionally.
Status CollectDataToRepository(const ProfileRequest& request,
                               ProfilerSession* profiler,
                               ProfileResponse* response) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/profiler/rpc/profiler_service_impl.cc", "CollectDataToRepository");

  response->set_empty_trace(true);
  // Read the profile data into xspace.
  XSpace xspace;
  TF_RETURN_IF_ERROR(profiler->CollectData(&xspace));
  xspace.add_hostnames(request.host_name());
  VLOG(3) << "Collected XSpace to repository.";
  response->set_empty_trace(IsEmpty(xspace));

  std::string log_dir_path =
      ProfilerJoinPath(request.repository_root(), request.session_id());
  VLOG(1) << "Creating " << log_dir_path;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(log_dir_path));

  std::string file_name = absl::StrCat(request.host_name(), ".", kXPlanePb);
  // Windows file names do not support colons.
  absl::StrReplaceAll({{":", "_"}}, &file_name);
  // Dumps profile data to <repository_root>/<run>/<host>_<port>.<kXPlanePb>
  std::string out_path = ProfilerJoinPath(log_dir_path, file_name);
  LOG(INFO) << "Collecting XSpace to repository: " << out_path;

  return WriteBinaryProto(Env::Default(), out_path, xspace);
}

class ProfilerServiceImpl : public grpc::ProfilerService::Service {
 public:
  ::grpc::Status Monitor(::grpc::ServerContext* ctx, const MonitorRequest* req,
                         MonitorResponse* response) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/profiler/rpc/profiler_service_impl.cc", "Monitor");

    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "unimplemented.");
  }

  ::grpc::Status Profile(::grpc::ServerContext* ctx, const ProfileRequest* req,
                         ProfileResponse* response) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/profiler/rpc/profiler_service_impl.cc", "Profile");

    VLOG(1) << "Received a profile request: " << req->DebugString();
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(req->opts());
    Status status = profiler->Status();
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    Env* env = Env::Default();
    for (uint64 i = 0; i < req->opts().duration_ms(); ++i) {
      env->SleepForMicroseconds(EnvTime::kMillisToMicros);
      if (ctx->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
      if (TF_PREDICT_FALSE(IsStopped(req->session_id()))) {
        mutex_lock lock(mutex_);
        stop_signals_per_session_.erase(req->session_id());
        break;
      }
    }

    status = CollectDataToRepository(*req, profiler.get(), response);
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    return ::grpc::Status::OK;
  }

  ::grpc::Status Terminate(::grpc::ServerContext* ctx,
                           const TerminateRequest* req,
                           TerminateResponse* response) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/profiler/rpc/profiler_service_impl.cc", "Terminate");

    mutex_lock lock(mutex_);
    stop_signals_per_session_[req->session_id()] = true;
    return ::grpc::Status::OK;
  }

 private:
  bool IsStopped(const std::string& session_id) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("session_id: \"" + session_id + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSprofiler_service_implDTcc mht_4(mht_4_v, 303, "", "./tensorflow/core/profiler/rpc/profiler_service_impl.cc", "IsStopped");

    mutex_lock lock(mutex_);
    auto it = stop_signals_per_session_.find(session_id);
    return it != stop_signals_per_session_.end() && it->second;
  }

  mutex mutex_;
  absl::flat_hash_map<std::string, bool> stop_signals_per_session_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace

std::unique_ptr<grpc::ProfilerService::Service> CreateProfilerService() {
  return absl::make_unique<ProfilerServiceImpl>();
}

}  // namespace profiler
}  // namespace tensorflow
