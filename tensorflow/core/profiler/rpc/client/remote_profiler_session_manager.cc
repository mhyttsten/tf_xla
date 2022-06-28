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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc() {
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

#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"

#include <cstddef>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

/*static*/ std::unique_ptr<RemoteProfilerSessionManager>
RemoteProfilerSessionManager::Create(
    const RemoteProfilerSessionManagerOptions& options,
    const ProfileRequest& request, tensorflow::Status& out_status,
    AddressResolver resolver) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "RemoteProfilerSessionManager::Create");

  VLOG(1) << "Creating a RemoteProfilerSessionManager.";
  auto session_manager = absl::WrapUnique(
      new RemoteProfilerSessionManager(options, request, resolver));
  out_status = session_manager->Init();
  if (!out_status.ok()) {
    return nullptr;
  }
  return session_manager;
}

RemoteProfilerSessionManager::RemoteProfilerSessionManager(
    RemoteProfilerSessionManagerOptions options, ProfileRequest request,
    AddressResolver resolver)
    : options_(options), request_(request) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "RemoteProfilerSessionManager::RemoteProfilerSessionManager");

  if (resolver) {
    resolver_ = resolver;
  } else {
    resolver_ = [](absl::string_view addr) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("addr: \"" + std::string(addr.data(), addr.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "lambda");
 return std::string(addr); };
  }
}

RemoteProfilerSessionManager::~RemoteProfilerSessionManager() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "RemoteProfilerSessionManager::~RemoteProfilerSessionManager");

  VLOG(2) << "Destroying RemoteProfilerSessionManager.";
}

Status RemoteProfilerSessionManager::Init() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "RemoteProfilerSessionManager::Init");

  mutex_lock lock(mutex_);
  VLOG(1) << "SessionManager initializing.";

  const absl::Time session_created_ts =
      absl::FromUnixNanos(options_.session_creation_timestamp_ns());
  const absl::Time deadline =
      session_created_ts +
      absl::Milliseconds(options_.max_session_duration_ms());

  LOG(INFO) << "Deadline set to " << deadline
            << " because max_session_duration_ms was "
            << options_.max_session_duration_ms()
            << " and session_creation_timestamp_ns was "
            << options_.session_creation_timestamp_ns() << " ["
            << session_created_ts << "]";

  // Prepare a list of clients.
  clients_.reserve(options_.service_addresses_size());

  ProfileRequest request = request_;
  for (auto& service_address : options_.service_addresses()) {
    std::string resolved_service_address = resolver_(service_address);
    request.set_host_name(resolved_service_address);

    // Creation also issues Profile RPC asynchronously.
    auto client = RemoteProfilerSession::Create(resolved_service_address,
                                                deadline, request);
    clients_.push_back(std::move(client));
  }

  LOG(INFO) << "Issued Profile gRPC to " << clients_.size() << " clients";
  return Status::OK();
}

std::vector<RemoteProfilerSessionManager::Response>
RemoteProfilerSessionManager::WaitForCompletion() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_managerDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.cc", "RemoteProfilerSessionManager::WaitForCompletion");

  mutex_lock lock(mutex_);
  std::vector<RemoteProfilerSessionManager::Response> remote_responses(
      clients_.size());

  for (int32_t idx = 0; idx < clients_.size(); ++idx) {
    auto& remote_response = remote_responses[idx];
    auto* client = clients_[idx].get();
    remote_response.profile_response =
        client->WaitForCompletion(remote_response.status);
    remote_response.service_address = std::string(client->GetServiceAddress());
  }
  return remote_responses;
}

}  // namespace profiler
}  // namespace tensorflow
