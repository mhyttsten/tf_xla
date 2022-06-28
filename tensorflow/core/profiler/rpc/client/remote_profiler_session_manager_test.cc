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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_manager_testDTcc() {
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
#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client_test_util.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::test::DurationApproxLess;
using ::tensorflow::profiler::test::DurationNear;
using ::tensorflow::profiler::test::StartServer;
using ::tensorflow::testing::TmpDir;
using Response = tensorflow::profiler::RemoteProfilerSessionManager::Response;

// Copied from capture_profile to not introduce a dependency.
ProfileRequest PopulateProfileRequest(
    absl::string_view repository_root, absl::string_view session_id,
    absl::string_view host_name,
    const RemoteProfilerSessionManagerOptions& options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("repository_root: \"" + std::string(repository_root.data(), repository_root.size()) + "\"");
   mht_0_v.push_back("session_id: \"" + std::string(session_id.data(), session_id.size()) + "\"");
   mht_0_v.push_back("host_name: \"" + std::string(host_name.data(), host_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSremote_profiler_session_manager_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/profiler/rpc/client/remote_profiler_session_manager_test.cc", "PopulateProfileRequest");

  constexpr uint64 kMaxEvents = 1000000;
  const absl::string_view kXPlanePb = "xplane.pb";
  ProfileRequest request;
  // TODO(b/169976117) Remove duration from request.
  request.set_duration_ms(options.profiler_options().duration_ms());
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  request.set_host_name(host_name.data(), host_name.size());
  // XPlane tool is only used by OSS profiler and safely ignored by TPU
  // profiler.
  request.add_tools(kXPlanePb.data(), kXPlanePb.size());
  *request.mutable_opts() = options.profiler_options();
  return request;
}

TEST(RemoteProfilerSessionManagerTest, Simple) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() =
      tensorflow::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Time approx_start = absl::Now();
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

TEST(RemoteProfilerSessionManagerTest, ExpiredDeadline) {
  absl::Duration duration = absl::Milliseconds(30);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() =
      tensorflow::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Duration grace = absl::Seconds(1);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  // This will create a deadline in the past.
  options.set_session_creation_timestamp_ns(0);

  absl::Time approx_start = absl::Now();
  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  EXPECT_THAT(elapsed, DurationNear(absl::Seconds(0)));
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(errors::IsDeadlineExceeded(responses.back().status));
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
}

TEST(RemoteProfilerSessionManagerTest, LongSession) {
  absl::Duration duration = absl::Seconds(3);
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() =
      tensorflow::ProfilerSession::DefaultOptions();
  options.mutable_profiler_options()->set_duration_ms(
      absl::ToInt64Milliseconds(duration));

  std::string service_address;
  auto server = StartServer(duration, &service_address);
  options.add_service_addresses(service_address);
  absl::Time approx_start = absl::Now();
  // Empirically determined value.
  absl::Duration grace = absl::Seconds(20);
  absl::Duration max_duration = duration + grace;
  options.set_max_session_duration_ms(absl::ToInt64Milliseconds(max_duration));
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(approx_start));

  ProfileRequest request =
      PopulateProfileRequest(TmpDir(), "session_id", service_address, options);
  Status status;
  auto sessions =
      RemoteProfilerSessionManager::Create(options, request, status);
  EXPECT_TRUE(status.ok());
  std::vector<Response> responses = sessions->WaitForCompletion();
  absl::Duration elapsed = absl::Now() - approx_start;
  ASSERT_EQ(responses.size(), 1);
  EXPECT_TRUE(responses.back().status.ok());
  EXPECT_TRUE(responses.back().profile_response->empty_trace());
  EXPECT_EQ(responses.back().profile_response->tool_data_size(), 0);
  EXPECT_THAT(elapsed, DurationApproxLess(max_duration));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
