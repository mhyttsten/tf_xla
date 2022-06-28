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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/distributed/service.h"

#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"

namespace xla {

DistributedRuntimeServiceImpl::DistributedRuntimeServiceImpl(
    const Options& options)
    : options_(options), session_id_(tensorflow::random::New64()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::DistributedRuntimeServiceImpl");

  nodes_.resize(options.num_nodes);
  local_topologies_.resize(options.num_nodes);
}

DistributedRuntimeServiceImpl::~DistributedRuntimeServiceImpl() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::~DistributedRuntimeServiceImpl");

  {
    absl::MutexLock lock(&mu_);
    state_ = State::kClosed;
    service_status_ =
        tensorflow::errors::FailedPrecondition("Service shutting down.");
    if (!stop_heartbeat_thread_.HasBeenNotified()) {
      stop_heartbeat_thread_.Notify();
    }
  }
}

// Steals the contents of `local_topologies`.
void BuildGlobalTopology(absl::Span<LocalTopologyProto> local_topologies,
                         GlobalTopologyProto* global_topology) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "BuildGlobalTopology");

  int next_global_device_id = 0;
  for (LocalTopologyProto& local : local_topologies) {
    for (DeviceProto& device : *local.mutable_devices()) {
      device.set_global_device_id(next_global_device_id++);
    }
    global_topology->add_nodes()->Swap(&local);
  }
}

xla::Status DistributedRuntimeServiceImpl::ValidateNodeId(int node_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::ValidateNodeId");

  if (node_id < 0) {
    return xla::InvalidArgument("Invalid node ID %d, must be non-negative",
                                node_id);
  }
  if (node_id >= options_.num_nodes) {
    return xla::FailedPrecondition(
        "Invalid node ID %d, must be in the range [0, %d)", node_id,
        options_.num_nodes);
  }
  return xla::Status::OK();
}

xla::Status DistributedRuntimeServiceImpl::ValidateSessionId(
    uint64_t session_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::ValidateSessionId");

  if (session_id != session_id_) {
    return xla::FailedPrecondition(
        "Session ID of request %llu does not match active session ID %llu",
        session_id, session_id_);
  }
  return xla::Status::OK();
}

::grpc::Status DistributedRuntimeServiceImpl::Connect(
    ::grpc::ServerContext* context, const ConnectRequest* request,
    ConnectResponse* response) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_5(mht_5_v, 268, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::Connect");

  VLOG(10) << "Connect " << request->DebugString();
  if (request->protocol_version() != DistributedRuntimeProtocolVersion()) {
    return ToGrpcStatus(xla::InvalidArgument("Invalid protocol version %d",
                                             request->protocol_version()));
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kInitializing) {
    // This most likely indicates that a client task was restarted but the
    // old master is still up. Clients should retry on failure.
    return ToGrpcStatus(tensorflow::errors::Aborted(
        "Connect() called when system is not initializing."));
  }
  int node_id = request->node_id();
  xla::Status status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  if (!nodes_[node_id].present) {
    nodes_[node_id].present = true;
    ++num_nodes_present_;
  }
  nodes_[node_id].client_id = request->client_id();

  auto all_nodes_present_or_duplicate_request = [&]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_6(mht_6_v, 295, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "lambda");

    mu_.AssertHeld();
    return num_nodes_present_ == nodes_.size() ||
           nodes_[node_id].client_id != request->client_id();
  };
  auto connect_timeout = absl::Milliseconds(request->timeout_milliseconds());
  if (!mu_.AwaitWithTimeout(
          absl::Condition(&all_nodes_present_or_duplicate_request),
          connect_timeout)) {
    nodes_[node_id].present = false;
    --num_nodes_present_;
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ", absl::FormatDuration(connect_timeout),
        " waiting for all nodes to call Connect()"));
  }

  if (nodes_[node_id].client_id != request->client_id()) {
    // This might happen either if two nodes are erroneously configured with the
    // same ID number, or it might happen if a task fails and is restarted
    // while we are waiting for nodes to connect. To elaborate on the second
    // scenario, it would look like this:
    // * a task calls Connect() with a particular node_id and client_id.
    // * the task is killed and restarted, or alternatively the client's RPC
    //   times out and it decides to retry.
    // * the task calls Connect() again with the same node_id and a different
    //   client_id.
    // In this scenario we take whichever client showed up most recently and
    // evict the client with an out-of-date client ID.
    return ToGrpcStatus(
        tensorflow::errors::Aborted("Duplicate node ID ", node_id));
  }

  if (node_id == 0) {
    state_ = State::kRunning;
    heartbeat_thread_.reset(options_.env->StartThread(
        tensorflow::ThreadOptions(), "pjrt_service_heartbeat",
        [this]() { HeartbeatLoop(); }));
  } else {
    auto running = [&]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_7(mht_7_v, 336, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "lambda");

      mu_.AssertHeld();
      return state_ == State::kRunning;
    };
    mu_.Await(absl::Condition(&running));
  }
  nodes_[node_id].last_heartbeat = absl::Now();
  response->set_session_id(session_id_);
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::Shutdown(
    ::grpc::ServerContext* context, const ShutdownRequest* request,
    ShutdownResponse* response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_8(mht_8_v, 352, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::Shutdown");

  VLOG(10) << "Shutdown " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "Shutdown() called when system is not running."));
  }
  int node_id = request->node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  ++num_nodes_shutting_down_;

  auto all_nodes_shutting_down = [&]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_9(mht_9_v, 376, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "lambda");

    mu_.AssertHeld();
    return num_nodes_shutting_down_ == nodes_.size() || !service_status_.ok();
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&all_nodes_shutting_down),
                            options_.shutdown_timeout)) {
    state_ = State::kClosed;
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ", absl::FormatDuration(options_.shutdown_timeout),
        " waiting for all nodes to call Shutdown()"));
  }
  state_ = State::kClosed;
  if (!stop_heartbeat_thread_.HasBeenNotified()) {
    stop_heartbeat_thread_.Notify();
  }
  if (!service_status_.ok()) {
    return ToGrpcStatus(service_status_);
  }
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::EnumerateDevices(
    ::grpc::ServerContext* context, const EnumerateDevicesRequest* request,
    EnumerateDevicesResponse* response) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_10(mht_10_v, 402, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::EnumerateDevices");

  VLOG(10) << "EnumerateDevices " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "EnumerateDevices() called when system is not running."));
  }
  int node_id = request->local_topology().node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  local_topologies_[node_id] = request->local_topology();
  ++num_topologies_present_;

  auto all_topologies_present = [&]() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_11(mht_11_v, 427, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "lambda");

    mu_.AssertHeld();
    return num_topologies_present_ == nodes_.size() || !service_status_.ok();
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&all_topologies_present),
                            options_.enumerate_devices_timeout)) {
    return ToGrpcStatus(tensorflow::errors::DeadlineExceeded(
        "Timed out after ",
        absl::FormatDuration(options_.enumerate_devices_timeout),
        " waiting for all nodes to call EnumerateDevices()"));
  }
  if (!service_status_.ok()) {
    return ToGrpcStatus(service_status_);
  }

  if (node_id == 0) {
    topology_.emplace();
    BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies_),
                        &*topology_);
    local_topologies_.clear();
  } else {
    auto topology_ready = [&]() -> bool {
      mu_.AssertHeld();
      return topology_.has_value();
    };
    mu_.Await(absl::Condition(&topology_ready));
  }
  *response->mutable_global_topology() = *topology_;
  return ::grpc::Status::OK;
}

::grpc::Status DistributedRuntimeServiceImpl::Heartbeat(
    ::grpc::ServerContext* context, const HeartbeatRequest* request,
    HeartbeatResponse* response) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_12(mht_12_v, 463, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::Heartbeat");

  VLOG(10) << "Heartbeat " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  absl::MutexLock lock(&mu_);
  if (state_ != State::kRunning) {
    if (!service_status_.ok()) {
      return ToGrpcStatus(service_status_);
    }
    return ToGrpcStatus(xla::FailedPrecondition(
        "Heartbeat() called when system is not running."));
  }
  int node_id = request->node_id();
  status = ValidateNodeId(node_id);
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  nodes_[node_id].last_heartbeat = absl::Now();
  return ::grpc::Status::OK;
}

void DistributedRuntimeServiceImpl::HeartbeatLoop() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_13(mht_13_v, 489, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::HeartbeatLoop");

  while (true) {
    stop_heartbeat_thread_.WaitForNotificationWithTimeout(
        options_.heartbeat_interval);
    VLOG(10) << "Checking heartbeats";
    if (stop_heartbeat_thread_.HasBeenNotified()) {
      VLOG(10) << "Heartbeat checking stopped.";
      return;
    }
    absl::Time now = absl::Now();
    absl::MutexLock lock(&mu_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
      // If we haven't heard from the node for a number of heartbeat intervals,
      // declare that we are unhealthy.
      VLOG(10) << "Node " << i
               << " last heartbeat: " << nodes_[i].last_heartbeat;
      if (nodes_[i].last_heartbeat +
              options_.max_missing_heartbeats * options_.heartbeat_interval <
          now) {
        LOG(INFO) << "Missed heartbeats from node " << i << ". Shutting down.";
        state_ = State::kClosed;
        service_status_ = tensorflow::errors::Aborted(
            "Shutting down due to missed heartbeat from task ", i);
        return;
      }
    }
  }
}

::grpc::Status DistributedRuntimeServiceImpl::KeyValueGet(
    ::grpc::ServerContext* context, const KeyValueGetRequest* request,
    KeyValueGetResponse* response) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_14(mht_14_v, 523, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::KeyValueGet");

  VLOG(10) << "KeyValueGet " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      if (!service_status_.ok()) {
        return ToGrpcStatus(service_status_);
      }
      return ToGrpcStatus(xla::FailedPrecondition(
          "KeyValueGet() called when system is not running."));
    }
  }
  return key_value_store_.Get(
      request->key(), absl::Milliseconds(request->timeout_milliseconds()),
      response->mutable_value());
}

::grpc::Status DistributedRuntimeServiceImpl::KeyValueSet(
    ::grpc::ServerContext* context, const KeyValueSetRequest* request,
    KeyValueSetResponse* response) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_15(mht_15_v, 549, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeServiceImpl::KeyValueSet");

  VLOG(10) << "KeyValueSet " << request->DebugString();
  xla::Status status = ValidateSessionId(request->session_id());
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  {
    absl::MutexLock lock(&mu_);
    if (state_ != State::kRunning) {
      if (!service_status_.ok()) {
        return ToGrpcStatus(service_status_);
      }
      return ToGrpcStatus(xla::FailedPrecondition(
          "KeyValueSet() called when system is not running; clients must call "
          "Connect() first"));
    }
  }
  return key_value_store_.Set(request->key(), request->value());
}

xla::StatusOr<std::unique_ptr<DistributedRuntimeService>>
DistributedRuntimeService::Get(
    const std::string& address,
    std::shared_ptr<::grpc::ServerCredentials> credentials,
    const DistributedRuntimeServiceImpl::Options& options) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("address: \"" + address + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_16(mht_16_v, 577, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeService::Get");

  auto service = absl::make_unique<DistributedRuntimeService>(options);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(address, credentials);
  VLOG(1) << "Distributed runtime service address " << address;
  builder.RegisterService(&service->impl_);
  service->server_ = builder.BuildAndStart();
  if (!service->server_) {
    return xla::Unknown("Failed to start RPC server");
  }
  LOG(INFO) << "Jax service listening on " << address;
  return service;
}

DistributedRuntimeService::DistributedRuntimeService(
    const DistributedRuntimeServiceImpl::Options& options)
    : impl_(options) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_17(mht_17_v, 596, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeService::DistributedRuntimeService");
}

DistributedRuntimeService::~DistributedRuntimeService() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_18(mht_18_v, 601, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeService::~DistributedRuntimeService");
 Shutdown(); }

void DistributedRuntimeService::Shutdown() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTcc mht_19(mht_19_v, 606, "", "./tensorflow/compiler/xla/pjrt/distributed/service.cc", "DistributedRuntimeService::Shutdown");

  if (server_) {
    LOG(INFO) << "Jax service shutting down";
    server_->Shutdown();
    server_->Wait();
  }
}

}  // namespace xla
