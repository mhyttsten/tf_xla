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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTh() {
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


#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"

namespace tensorflow {
namespace profiler {

// Note that tensorflow/tools/def_file_filter/symbols_pybind.txt is incompatible
// with absl::string_view.
Status ProfileGrpc(const std::string& service_address,
                   const ProfileRequest& request, ProfileResponse* response);

Status NewSessionGrpc(const std::string& service_address,
                      const NewProfileSessionRequest& request,
                      NewProfileSessionResponse* response);

Status MonitorGrpc(const std::string& service_address,
                   const MonitorRequest& request, MonitorResponse* response);

class RemoteProfilerSession {
 public:
  // Creates an instance and starts a remote profiling session immediately.
  // This is a non-blocking call and does not wait for a response.
  // Response must outlive the instantiation.
  static std::unique_ptr<RemoteProfilerSession> Create(
      const std::string& service_address, absl::Time deadline,
      const ProfileRequest& profile_request);

  // Not copyable or movable.
  RemoteProfilerSession(const RemoteProfilerSession&) = delete;
  RemoteProfilerSession& operator=(const RemoteProfilerSession&) = delete;

  ~RemoteProfilerSession();

  absl::string_view GetServiceAddress() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSprofiler_clientDTh mht_0(mht_0_v, 228, "", "./tensorflow/core/profiler/rpc/client/profiler_client.h", "GetServiceAddress");
 return service_address_; }

  // Blocks until a response has been received or until deadline expiry,
  // whichever is first. Subsequent calls after the first will yield nullptr and
  // an error status.
  std::unique_ptr<ProfileResponse> WaitForCompletion(Status& out_status);

 private:
  explicit RemoteProfilerSession(const std::string& service_addr,
                                 absl::Time deadline,
                                 const ProfileRequest& profile_request);

  // Starts a remote profiling session. This is a non-blocking call.
  // Will be called exactly once during instantiation.
  // RPC will write to response.profile_response eagerly. However, since
  // response.status requires a conversion from grpc::Status, it can only be
  //  evaluated lazily at WaitForCompletion() time.
  void ProfileAsync();

  Status status_on_completion_;
  std::unique_ptr<ProfileResponse> response_;
  // Client address and connection attributes.
  std::string service_address_;
  std::unique_ptr<grpc::ProfilerService::Stub> stub_;
  absl::Time deadline_;
  ::grpc::ClientContext grpc_context_;
  std::unique_ptr<::grpc::ClientAsyncResponseReader<ProfileResponse>> rpc_;
  ::grpc::Status grpc_status_ = ::grpc::Status::OK;

  // Asynchronous completion queue states.
  ::grpc::CompletionQueue cq_;

  ProfileRequest profile_request_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
