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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc() {
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

/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/profiler_analysis.pb.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"
#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::RemoteProfilerSessionManager;
using Response = ::tensorflow::profiler::RemoteProfilerSessionManager::Response;

constexpr uint64 kMaxEvents = 1000000;
const absl::string_view kXPlanePb = "xplane.pb";

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level,
                                      bool timestamp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "PopulateMonitorRequest");

  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  request.set_timestamp(timestamp);
  return request;
}

ProfileRequest PopulateProfileRequest(
    absl::string_view repository_root, absl::string_view session_id,
    absl::string_view host_name,
    const RemoteProfilerSessionManagerOptions& options) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("repository_root: \"" + std::string(repository_root.data(), repository_root.size()) + "\"");
   mht_1_v.push_back("session_id: \"" + std::string(session_id.data(), session_id.size()) + "\"");
   mht_1_v.push_back("host_name: \"" + std::string(host_name.data(), host_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "PopulateProfileRequest");

  ProfileRequest request;
  // TODO(b/169976117) Remove duration from request.
  request.set_duration_ms(options.profiler_options().duration_ms());
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  request.set_host_name(host_name.data(), host_name.size());
  // These tools are only used by TPU profiler.
  request.add_tools("trace_viewer");
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("memory_viewer");
  request.add_tools("memory_profile");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  request.add_tools("tensorflow_stats");
  // XPlane tool is only used by OSS profiler and safely ignored by TPU
  // profiler.
  request.add_tools(kXPlanePb.data(), kXPlanePb.size());
  *request.mutable_opts() = options.profiler_options();
  return request;
}

NewProfileSessionRequest PopulateNewProfileSessionRequest(
    absl::string_view repository_root, absl::string_view session_id,
    const RemoteProfilerSessionManagerOptions& opts) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("repository_root: \"" + std::string(repository_root.data(), repository_root.size()) + "\"");
   mht_2_v.push_back("session_id: \"" + std::string(session_id.data(), session_id.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "PopulateNewProfileSessionRequest");

  NewProfileSessionRequest request;
  std::vector<absl::string_view> parts =
      absl::StrSplit(opts.service_addresses(0), ':');
  DCHECK(!parts.empty());

  *request.mutable_request() =
      PopulateProfileRequest(repository_root, session_id, parts[0], opts);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  for (const auto& hostname : opts.service_addresses()) {
    request.add_hosts(hostname);
  }
  return request;
}

inline bool ShouldRetryTracing(Status status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "ShouldRetryTracing");

  return status.code() == error::Code::UNAVAILABLE ||
         status.code() == error::Code::ALREADY_EXISTS ||
         // When auto-reconnecting to a remote TensorFlow worker after it
         // restarts, gRPC can return an UNKNOWN error code with a "Stream
         // removed" error message. This should not be treated as an
         // unrecoverable error.
         (status.code() == error::Code::UNKNOWN &&
          status.error_message() == "Stream removed");
}

Status Profile(const std::string& repository_root,
               const std::string& session_id,
               const RemoteProfilerSessionManagerOptions& opts) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("repository_root: \"" + repository_root + "\"");
   mht_4_v.push_back("session_id: \"" + session_id + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_4(mht_4_v, 304, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "Profile");

  Status status;
  // Host name will be overwritten by RemoteProfilerSessionManager later.
  ProfileRequest request = PopulateProfileRequest(repository_root, session_id,
                                                  /*host_name=*/"", opts);
  auto session = RemoteProfilerSessionManager::Create(opts, request, status);
  TF_RETURN_IF_ERROR(status);
  // Expect one or more service addresses.
  DCHECK_GT(opts.service_addresses_size(), 0);
  std::vector<Response> responses = session->WaitForCompletion();
  // Expect responses to have the same size as clients.
  DCHECK_EQ(responses.size(), opts.service_addresses_size());

  bool has_trace_data = false;
  for (const auto& client_response : responses) {
    ProfileResponse& response = *client_response.profile_response;
    if (response.empty_trace()) {
      LOG(WARNING) << "No trace event is collected from "
                   << client_response.service_address;
    } else {
      has_trace_data = true;
      // If server side returns tool data in the response, saves that into the
      // repository. This improves backward compatibility by reducing assumption
      // of what server side does.
      TF_RETURN_IF_ERROR(SaveProfile(repository_root, session_id,
                                     client_response.service_address, response,
                                     &std::cout));
    }
    if (!client_response.status.ok()) {
      LOG(WARNING) << client_response.service_address << " returned "
                   << client_response.status;
    }
  }

  if (!has_trace_data) {
    return Status(error::Code::UNAVAILABLE,
                  "No trace event was collected because there were no responses"
                  " from clients or the responses did not have trace data.");
  }
  return Status::OK();
}

// Start a new profiling session that include all the hosts included in
// hostnames, for the time interval of duration_ms. Possibly save the profiling
// result in the directory specified by repository_root and session_id.
Status NewSession(absl::string_view repository_root,
                  absl::string_view session_id,
                  const RemoteProfilerSessionManagerOptions& opts) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("repository_root: \"" + std::string(repository_root.data(), repository_root.size()) + "\"");
   mht_5_v.push_back("session_id: \"" + std::string(session_id.data(), session_id.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_5(mht_5_v, 356, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "NewSession");

  NewProfileSessionRequest request =
      PopulateNewProfileSessionRequest(repository_root, session_id, opts);
  NewProfileSessionResponse response;
  TF_RETURN_IF_ERROR(
      NewSessionGrpc(opts.service_addresses(0), request, &response));

  std::cout << "Profile session succeed for host(s):"
            << absl::StrJoin(opts.service_addresses(), ",") << std::endl;
  if (response.empty_trace()) {
    return errors::Unavailable("No trace event is collected");
  }
  return Status::OK();
}

}  // namespace

Status Trace(const std::string& logdir, int num_tracing_attempts,
             RemoteProfilerSessionManagerOptions& opts,
             bool is_cloud_tpu_session) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("logdir: \"" + logdir + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_6(mht_6_v, 379, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "Trace");

  DCHECK_GT(opts.profiler_options().duration_ms(), 0);
  DCHECK(!opts.service_addresses().empty());

  // Use the current timestamp as the run name.
  std::string session_id = GetCurrentTimeStampAsString();
  std::string repository_root = GetTensorBoardProfilePluginDir(logdir);
  auto duration_ms = opts.profiler_options().duration_ms();
  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));

  Status status;
  int remaining_attempts = num_tracing_attempts;
  while (true) {
    auto start_timestamp = absl::Now() + absl::Milliseconds(opts.delay_ms());
    opts.mutable_profiler_options()->set_start_timestamp_ns(
        absl::ToUnixNanos(start_timestamp));
    LOG(INFO) << "Profiler delay_ms was " << opts.delay_ms()
              << ", start_timestamp_ns set to "
              << opts.profiler_options().start_timestamp_ns() << " ["
              << start_timestamp << "]";

    std::cout << "Starting to trace for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << --remaining_attempts << std::endl;

    if (is_cloud_tpu_session) {
      status = NewSession(repository_root, session_id, opts);
    } else {
      status = Profile(repository_root, session_id, opts);
    }
    if (remaining_attempts <= 0 || status.ok() || !ShouldRetryTracing(status))
      break;
    std::cout << "No trace event is collected. Automatically retrying.\n"
              << std::endl;
  }

  if (ShouldRetryTracing(status)) {
    std::cout << "No trace event is collected after " << num_tracing_attempts
              << " attempt(s). "
              << "Perhaps, you want to try again (with more attempts?).\n"
              << "Tip: increase number of attempts with --num_tracing_attempts."
              << std::endl;
  }
  return status;
}

Status Monitor(const std::string& service_addr, int duration_ms,
               int monitoring_level, bool display_timestamp,
               std::string* result) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("service_addr: \"" + service_addr + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_7(mht_7_v, 430, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "Monitor");

  MonitorRequest request =
      PopulateMonitorRequest(duration_ms, monitoring_level, display_timestamp);
  MonitorResponse response;
  TF_RETURN_IF_ERROR(MonitorGrpc(service_addr, request, &response));
  *result = response.data();
  return Status::OK();
}

Status ExportToTensorBoard(const XSpace& xspace, const std::string& logdir) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("logdir: \"" + logdir + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPScapture_profileDTcc mht_8(mht_8_v, 443, "", "./tensorflow/core/profiler/rpc/client/capture_profile.cc", "ExportToTensorBoard");

  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));

  ProfileResponse response;
  ProfileRequest request = PopulateProfileRequest(
      GetTensorBoardProfilePluginDir(logdir), GetCurrentTimeStampAsString(),
      port::Hostname(), /*options=*/{});
  TF_RETURN_IF_ERROR(
      ConvertXSpaceToProfileResponse(xspace, request, &response));
  std::stringstream ss;  // Record LOG messages.
  TF_RETURN_IF_ERROR(SaveProfile(request.repository_root(),
                                 request.session_id(), request.host_name(),
                                 response, &ss));
  LOG(INFO) << ss.str();
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
