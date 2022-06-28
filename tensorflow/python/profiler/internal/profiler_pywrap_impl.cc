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
class MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"

namespace tensorflow {
namespace profiler {
namespace pywrap {

namespace {

using ::tensorflow::RemoteProfilerSessionManagerOptions;

// Profiler gives grace after profiling duration to terminate.
constexpr absl::Duration kMinSessionGraceTime = absl::Seconds(60);

tensorflow::Status ValidateHostPortPair(absl::string_view host_port) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("host_port: \"" + std::string(host_port.data(), host_port.size()) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_0(mht_0_v, 221, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "ValidateHostPortPair");

  tensorflow::uint32 port;
  std::vector<absl::string_view> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      absl::StrContains(parts[0], "/") || parts[0].empty()) {
    return tensorflow::errors::InvalidArgument(
        "Could not interpret \"", host_port, "\" as a host-port pair.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateOptions(
    const RemoteProfilerSessionManagerOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_1(mht_1_v, 238, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "ValidateOptions");

  if (options.service_addresses().empty()) {
    return tensorflow::errors::InvalidArgument("No service address provided.");
  }

  if (options.profiler_options().duration_ms() == 0) {
    return tensorflow::errors::InvalidArgument(
        "duration_ms must be greater than zero.");
  }

  for (absl::string_view host_port : options.service_addresses()) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(host_port));
  }

  if (options.max_session_duration_ms() <
      options.profiler_options().duration_ms()) {
    return tensorflow::errors::InvalidArgument(
        "The maximum profiling session duration must be greater than or equal "
        "to the local profiler duration.");
  }

  return tensorflow::Status::OK();
}

// Receives a comma delimited list of service_addresses and adds them to
// RemoteProfilerSessionManagerOptions::service_addresses.
void AddServiceAddresses(absl::string_view service_addresses,
                         RemoteProfilerSessionManagerOptions* options) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("service_addresses: \"" + std::string(service_addresses.data(), service_addresses.size()) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_2(mht_2_v, 269, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "AddServiceAddresses");

  for (absl::string_view server : absl::StrSplit(service_addresses, ',')) {
    options->add_service_addresses(server.data(), server.size());
  }
}

// Sets gRPC deadline to a grace period based on the profiling duration.
void UpdateMaxSessionDuration(RemoteProfilerSessionManagerOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_3(mht_3_v, 279, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "UpdateMaxSessionDuration");

  auto local_profiler_duration = options.profiler_options().duration_ms();
  auto session_creation_ts = options.session_creation_timestamp_ns();
  auto requested_start_ts = options.profiler_options().start_timestamp_ns();
  // User only needs to set maximal session duration if the profiling duration
  // is bounded.
  DCHECK_GT(local_profiler_duration, 0);
  VLOG(3) << "duration_ms was given as " << local_profiler_duration;
  // Max session duration is the profiling session with grace time.
  auto profile_duration = std::max(
      kMinSessionGraceTime, absl::Milliseconds(local_profiler_duration) * 2);
  absl::Duration delay_duration;
  // When requested start timestamp is 0, profiling starts immediately.
  if (requested_start_ts > 0) {
    delay_duration =
        absl::Nanoseconds(requested_start_ts - session_creation_ts);
  }

  auto max_session_duration = profile_duration + delay_duration;
  options.set_max_session_duration_ms(
      absl::ToInt64Milliseconds(max_session_duration));
  VLOG(1) << "max_session_duration set to " << max_session_duration;
}

// Takes profiler options in absl::flat_hash_map and returns a
// RemoteProfilerSessionManagerOptions.
RemoteProfilerSessionManagerOptions GetOptionsLocked(
    absl::string_view logdir,
    const absl::flat_hash_map<std::string, absl::variant<int>>& opts) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("logdir: \"" + std::string(logdir.data(), logdir.size()) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_4(mht_4_v, 311, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "GetOptionsLocked");

  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() =
      tensorflow::ProfilerSession::DefaultOptions();
  // Store a timestamp of when this session was created. This will be the basis
  // of gRPC deadline afterwards.
  auto now = absl::Now();
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(now));
  VLOG(2) << "set_session_creation_timestamp_ns set to "
          << options.session_creation_timestamp_ns() << " [" << now << "]";

  // Set the path of where to store XSpaces.
  options.mutable_profiler_options()->set_repository_path(logdir.data(),
                                                          logdir.size());
  VLOG(2) << "repository_path set to "
          << options.profiler_options().repository_path();

  for (const auto& kw : opts) {
    absl::string_view key = kw.first;
    if (key == "host_tracer_level") {
      int value = absl::get<int>(kw.second);
      options.mutable_profiler_options()->set_host_tracer_level(value);
      VLOG(1) << "host_tracer_level set to " << value;
    } else if (key == "device_tracer_level") {
      int value = absl::get<int>(kw.second);
      options.mutable_profiler_options()->set_device_tracer_level(value);
      VLOG(1) << "device_tracer_level set to " << value;
    } else if (key == "python_tracer_level") {
      int value = absl::get<int>(kw.second);
      options.mutable_profiler_options()->set_python_tracer_level(value);
      VLOG(1) << "python_tracer_level set to " << value;
    } else if (key == "delay_ms") {
      int value = absl::get<int>(kw.second);
      options.set_delay_ms(value);
      VLOG(1) << "delay_ms was set to " << value;
    } else {
      LOG(WARNING) << "Unrecognised key: " << key;
    }
  }

  return options;
}

RemoteProfilerSessionManagerOptions GetOptionsLocked(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    int32_t duration_ms,
    const absl::flat_hash_map<std::string, absl::variant<int>>& opts,
    bool* is_cloud_tpu_session) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("service_addresses: \"" + std::string(service_addresses.data(), service_addresses.size()) + "\"");
   mht_5_v.push_back("logdir: \"" + std::string(logdir.data(), logdir.size()) + "\"");
   mht_5_v.push_back("worker_list: \"" + std::string(worker_list.data(), worker_list.size()) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_5(mht_5_v, 365, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "GetOptionsLocked");

  auto options = GetOptionsLocked(logdir, opts);

  // Remote profiling does not support any use cases where the following options
  // are set by `opts`. e.g. `opts['service_addrs']` will not happen.
  DCHECK(options.service_addresses().empty());
  // In remote profiling, duration is always passed by value explicitly and not
  // set in opts.
  DCHECK_EQ(options.profiler_options().duration_ms(), 0);
  // Because duration_ms is not set from opts, it follows that
  // max_session_duration_ms must be unset as well.
  DCHECK_EQ(options.max_session_duration_ms(), 0);

  // Worker_list is only used for TensorBoard TPU capture cases. For a TPU
  // cluster, service_address is the Master, which can already be found in the
  // list of workers. These sessions will be used with the ProfileAnalysis
  // service.
  *is_cloud_tpu_session = !worker_list.empty();
  AddServiceAddresses(*is_cloud_tpu_session ? worker_list : service_addresses,
                      &options);

  // Set local profiler duration and profiler session durations.
  options.mutable_profiler_options()->set_include_dataset_ops(
      include_dataset_ops);
  options.mutable_profiler_options()->set_duration_ms(duration_ms);
  UpdateMaxSessionDuration(options);

  for (int idx = 0; idx < options.service_addresses_size(); ++idx) {
    VLOG(1) << "service_addr " << idx << " set to "
            << options.service_addresses(idx);
  }
  VLOG(1) << "include_dataset_ops set to " << include_dataset_ops;
  VLOG(1) << "duration_ms set to " << duration_ms;

  return options;
}

}  // namespace

tensorflow::Status Trace(
    const char* service_addr, const char* logdir, const char* worker_list,
    bool include_dataset_ops, int duration_ms, int num_tracing_attempts,
    const absl::flat_hash_map<std::string, absl::variant<int>>& options) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("service_addr: \"" + (service_addr == nullptr ? std::string("nullptr") : std::string((char*)service_addr)) + "\"");
   mht_6_v.push_back("logdir: \"" + (logdir == nullptr ? std::string("nullptr") : std::string((char*)logdir)) + "\"");
   mht_6_v.push_back("worker_list: \"" + (worker_list == nullptr ? std::string("nullptr") : std::string((char*)worker_list)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_6(mht_6_v, 413, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "Trace");

  // TPU capture is true if the user sets worker_list.
  bool is_cloud_tpu_session = false;
  RemoteProfilerSessionManagerOptions opts =
      GetOptionsLocked(service_addr, logdir, worker_list, include_dataset_ops,
                       duration_ms, options, &is_cloud_tpu_session);
  TF_RETURN_IF_ERROR(ValidateOptions(opts));

  {
    TF_RETURN_IF_ERROR(tensorflow::profiler::Trace(logdir, num_tracing_attempts,
                                                   opts, is_cloud_tpu_session));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Monitor(const char* service_addr, int duration_ms,
                           int monitoring_level, bool display_timestamp,
                           tensorflow::string* result) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("service_addr: \"" + (service_addr == nullptr ? std::string("nullptr") : std::string((char*)service_addr)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_7(mht_7_v, 434, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "Monitor");

  TF_RETURN_IF_ERROR(ValidateHostPortPair(service_addr));
  {
    TF_RETURN_IF_ERROR(tensorflow::profiler::Monitor(
        service_addr, duration_ms, monitoring_level, display_timestamp,
        result));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ProfilerSessionWrapper::Start(
    const char* logdir,
    const absl::flat_hash_map<std::string, absl::variant<int>>& options) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("logdir: \"" + (logdir == nullptr ? std::string("nullptr") : std::string((char*)logdir)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_8(mht_8_v, 450, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "ProfilerSessionWrapper::Start");

  auto opts = GetOptionsLocked(logdir, options);
  session_ = tensorflow::ProfilerSession::Create(opts.profiler_options());
  logdir_ = logdir;
  return session_->Status();
}

tensorflow::Status ProfilerSessionWrapper::Stop(tensorflow::string* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_9(mht_9_v, 460, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "ProfilerSessionWrapper::Stop");

  if (session_ != nullptr) {
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status = session_->CollectData(&xspace);
    session_.reset();
    tensorflow::profiler::ConvertXSpaceToTraceEventsString(xspace, result);
    TF_RETURN_IF_ERROR(status);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ProfilerSessionWrapper::ExportToTensorBoard() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSprofilerPSinternalPSprofiler_pywrap_implDTcc mht_10(mht_10_v, 474, "", "./tensorflow/python/profiler/internal/profiler_pywrap_impl.cc", "ProfilerSessionWrapper::ExportToTensorBoard");

  if (!session_ || logdir_.empty()) {
    return Status::OK();
  }
  tensorflow::profiler::XSpace xspace;
  tensorflow::Status status;
  status = session_->CollectData(&xspace);
  xspace.add_hostnames(tensorflow::port::Hostname());
  session_.reset();
  status = tensorflow::profiler::ExportToTensorBoard(xspace, logdir_);
  return status;
}

}  // namespace pywrap
}  // namespace profiler
}  // namespace tensorflow
