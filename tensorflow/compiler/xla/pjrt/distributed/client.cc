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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc() {
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

/* Copyright 2020 Google LLC

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

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <random>
#include <string>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"

namespace xla {
class DistributedRuntimeClientImpl : public DistributedRuntimeClient {
 public:
  DistributedRuntimeClientImpl(std::shared_ptr<::grpc::Channel> channel,
                               const Options& options);
  explicit DistributedRuntimeClientImpl(
      std::shared_ptr<::grpc::Channel> channel)
      : DistributedRuntimeClientImpl(channel, Options()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl");
}
  ~DistributedRuntimeClientImpl() override;

  xla::Status Connect() override;
  xla::Status Shutdown() override;
  xla::Status EnumerateDevices(const LocalTopologyProto& local_topology,
                               GlobalTopologyProto* global_topology) override;
  xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) override;
  xla::Status KeyValueSet(std::string key, std::string value) override;

 private:
  // Entry point for the heartbeat thread.
  void HeartbeatLoop();

  const std::unique_ptr<grpc::DistributedRuntimeService::Stub> stub_;
  const DistributedRuntimeClient::Options options_;

  // Possible states of the client.
  // The only legal transitions are downwards in the order below. i.e., there is
  // no way to reopen a closed client.
  enum class State {
    // The client has not yet connected to the server, i.e., had a Connect()
    // RPC succeed.
    kNotConnected,

    // The client is connected to the server and as far as we are aware the
    // connection is healthy.
    kConnected,

    // The client is in the process of shutting down, i.e., Shutdown() has been
    // called.
    kShuttingDown,

    // The client has shut down its server connection, either due to an error
    // or due to an explicit shutdown.
    kClosed,
  };

  static absl::string_view StateToString(State state);

  // state_ is protected by a mutex because the heartbeat thread needs to look
  // at it.
  absl::Mutex mu_;
  State state_ ABSL_GUARDED_BY(mu_) = State::kNotConnected;

  // A unique session ID, assigned by the server during Connect().
  uint64_t session_id_;

  // Notification that tells the heartbeat thread to stop running.
  absl::Notification stop_heartbeats_;

  // Thread responsible for performing heartbeats.
  std::unique_ptr<tensorflow::Thread> heartbeat_thread_;
};

DistributedRuntimeClientImpl::DistributedRuntimeClientImpl(
    std::shared_ptr<::grpc::Channel> channel, const Options& options)
    : stub_(grpc::DistributedRuntimeService::NewStub(std::move(channel))),
      options_(options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_1(mht_1_v, 272, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::DistributedRuntimeClientImpl");
}

DistributedRuntimeClientImpl::~DistributedRuntimeClientImpl() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_2(mht_2_v, 277, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::~DistributedRuntimeClientImpl");

  bool connected;
  {
    absl::MutexLock lock(&mu_);
    connected = (state_ == State::kConnected);
  }
  if (connected) {
    if (options_.shutdown_on_destruction) {
      Status status = Shutdown();
      if (!status.ok()) {
        LOG(WARNING) << "PJRT shutdown failed: " << status;
      }
    } else {
      if (!stop_heartbeats_.HasBeenNotified()) {
        stop_heartbeats_.Notify();
      }
    }
  }
}

/*static*/ absl::string_view DistributedRuntimeClientImpl::StateToString(
    State state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_3(mht_3_v, 301, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::StateToString");

  switch (state) {
    case State::kNotConnected:
      return "kNotConnected";
    case State::kConnected:
      return "kConnected";
    case State::kShuttingDown:
      return "kShuttingDown";
    case State::kClosed:
      return "kClosed";
  }
}

xla::Status DistributedRuntimeClientImpl::Connect() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_4(mht_4_v, 317, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::Connect");

  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kNotConnected) {
      return xla::FailedPrecondition("Connect() called when client in state %s",
                                     StateToString(state_));
    }
  }
  ConnectRequest request;
  request.set_protocol_version(DistributedRuntimeProtocolVersion());
  request.set_timeout_milliseconds(
      absl::ToInt64Milliseconds(options_.rpc_timeout) / 2);
  request.set_node_id(options_.node_id);
  VLOG(10) << "Connect: " << request.DebugString();
  ConnectResponse response;
  ::grpc::Status status;
  absl::Time deadline = absl::Now() + options_.init_timeout;
  int attempt = 0;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  do {
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
    request.set_client_id(tensorflow::random::New64());
    response.Clear();
    status = stub_->Connect(&ctx, request, &response);
    if (!status.ok()) {
      VLOG(1) << "Connect failed() with status: " << FromGrpcStatus(status);
      if (attempt % 10 == 0) {
        LOG(INFO) << "Connect failed() with status: " << FromGrpcStatus(status);
      }
      // Exponential backoff with jitter. Note we will retry for `init_timeout`
      // time in total; the `14` here corresponds to an ~16s maximum interval
      // between connection attempts.
      int backoff = 1 << std::min(14, attempt);
      absl::SleepFor(absl::Milliseconds(backoff * distribution(generator)));
    }
    ++attempt;
  } while (!status.ok() && absl::Now() < deadline);
  if (!status.ok()) {
    LOG(ERROR) << "Connect() failed after " << attempt << " retries in "
               << options_.init_timeout
               << "; most recent failure status: " << FromGrpcStatus(status);
    return tensorflow::errors::DeadlineExceeded(
        absl::StrFormat("Connect() timed out after %s with %d attempts. Most "
                        "recent failure was: %s",
                        absl::FormatDuration(options_.init_timeout), attempt,
                        FromGrpcStatus(status).ToString()));
  }
  VLOG(10) << "Connect() response: " << response.DebugString();
  {
    absl::MutexLock lock(&mu_);
    state_ = State::kConnected;
  }
  session_id_ = response.session_id();

  heartbeat_thread_.reset(options_.env->StartThread(
      tensorflow::ThreadOptions(), "pjrt_distributed_heartbeat",
      [this]() { HeartbeatLoop(); }));
  LOG(INFO) << "Connected to distributed JAX controller";
  return xla::Status::OK();
}

xla::Status DistributedRuntimeClientImpl::EnumerateDevices(
    const LocalTopologyProto& local_topology,
    GlobalTopologyProto* global_topology) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_5(mht_5_v, 386, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::EnumerateDevices");

  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "EnumerateDevices() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
  EnumerateDevicesRequest request;
  request.set_session_id(session_id_);
  *request.mutable_local_topology() = local_topology;
  request.mutable_local_topology()->set_node_id(options_.node_id);

  VLOG(10) << "EnumerateDevices: " << request.DebugString();
  EnumerateDevicesResponse response;
  ::grpc::Status status = stub_->EnumerateDevices(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  VLOG(10) << "EnumerateDevices() response: " << response.DebugString();
  response.mutable_global_topology()->Swap(global_topology);
  return xla::Status::OK();
}

xla::Status DistributedRuntimeClientImpl::Shutdown() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_6(mht_6_v, 416, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::Shutdown");

  LOG(INFO) << "Waiting for all distributed JAX tasks to shut down.";
  ::grpc::ClientContext ctx;
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "Shutdown() called when client not connected.");
    }
    state_ = State::kShuttingDown;
  }
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.shutdown_timeout));
  ShutdownRequest request;
  request.set_session_id(session_id_);
  VLOG(10) << "Shutdown: " << request.DebugString();
  ShutdownResponse response;
  ::grpc::Status status = stub_->Shutdown(&ctx, request, &response);
  LOG(INFO) << "Distributed task shutdown result: " << FromGrpcStatus(status);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  if (!stop_heartbeats_.HasBeenNotified()) {
    stop_heartbeats_.Notify();
  }
  VLOG(10) << "Shutdown() response: " << response.DebugString();
  absl::MutexLock lock(&mu_);
  state_ = State::kClosed;
  return xla::Status::OK();
}

xla::StatusOr<std::string> DistributedRuntimeClientImpl::BlockingKeyValueGet(
    std::string key, absl::Duration timeout) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_7(mht_7_v, 452, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::BlockingKeyValueGet");

  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "BlockingKeyValueGet() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  KeyValueGetRequest request;
  request.set_session_id(session_id_);
  request.set_key(std::move(key));
  timeout = std::min(timeout, absl::Minutes(10));  // Avoid overflow
  request.set_timeout_milliseconds(timeout / absl::Milliseconds(1));
  VLOG(10) << "BlockingKeyValueGet: " << request.DebugString();
  KeyValueGetResponse response;
  ::grpc::Status status = stub_->KeyValueGet(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.value();
}

xla::Status DistributedRuntimeClientImpl::KeyValueSet(std::string key,
                                                      std::string value) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("key: \"" + key + "\"");
   mht_8_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_8(mht_8_v, 483, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::KeyValueSet");

  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kConnected) {
      return xla::FailedPrecondition(
          "KeyValueSet() called when client not connected.");
    }
  }
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + options_.rpc_timeout));
  KeyValueSetRequest request;
  request.set_session_id(session_id_);
  request.set_key(std::move(key));
  request.set_value(std::move(value));
  VLOG(10) << "KeyValueSet: " << request.DebugString();
  KeyValueSetResponse response;
  ::grpc::Status status = stub_->KeyValueSet(&ctx, request, &response);
  return FromGrpcStatus(status);
}

void DistributedRuntimeClientImpl::HeartbeatLoop() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTcc mht_9(mht_9_v, 507, "", "./tensorflow/compiler/xla/pjrt/distributed/client.cc", "DistributedRuntimeClientImpl::HeartbeatLoop");

  int num_missing_heartbeats = 0;
  while (true) {
    stop_heartbeats_.WaitForNotificationWithTimeout(
        options_.heartbeat_interval);
    if (stop_heartbeats_.HasBeenNotified()) {
      return;
    }

    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    ctx.set_deadline(
        absl::ToChronoTime(absl::Now() + options_.heartbeat_interval));
    HeartbeatRequest request;
    request.set_session_id(session_id_);
    request.set_node_id(options_.node_id);
    VLOG(10) << "Heartbeat: " << request.DebugString();
    HeartbeatResponse response;
    ::grpc::Status status = stub_->Heartbeat(&ctx, request, &response);
    if (status.ok()) {
      VLOG(10) << "Heartbeat ok";
      num_missing_heartbeats = 0;
    } else {
      ++num_missing_heartbeats;
      VLOG(10) << "Heartbeat error, "
               << options_.max_missing_heartbeats - num_missing_heartbeats
               << " tries left: " << status.error_message();
      bool is_transient_error =
          (status.error_code() == ::grpc::StatusCode::DEADLINE_EXCEEDED ||
           status.error_code() == ::grpc::StatusCode::UNAVAILABLE);
      if (!stop_heartbeats_.HasBeenNotified() &&
          (!is_transient_error ||
           num_missing_heartbeats >= options_.max_missing_heartbeats)) {
        // If we are shutting down, missed heartbeats are benign: they may
        // simply mean that the server has shut down already before it saw
        // the heartbeat request.
        absl::MutexLock lock(&mu_);
        if (state_ != State::kShuttingDown) {
          options_.missed_heartbeat_callback(FromGrpcStatus(status),
                                             !is_transient_error);
        }
        return;
      }
    }
  }
}

std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options) {
  return std::make_unique<xla::DistributedRuntimeClientImpl>(channel, options);
}
}  // namespace xla
