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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTh() {
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


#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/key_value_store.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

typedef int NodeId;

class DistributedRuntimeServiceImpl final
    : public grpc::DistributedRuntimeService::Service {
 public:
  struct Options {
    // Number of nodes in the job. Mandatory. Must be non-negative.
    int num_nodes = -1;

    tensorflow::Env* env = tensorflow::Env::Default();

    // Interval at which the service should check for missed heartbeat RPCs
    // from the clients.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // Number of heartbeats that a client may miss in a row before the
    // coordinator concludes that a client has vanished.
    int max_missing_heartbeats = 10;

    // How long should we wait for all clients to call EnumerateDevices() before
    // giving up?
    absl::Duration enumerate_devices_timeout = absl::Seconds(60);

    // How long should we wait for all clients to call Shutdown() before giving
    // up and returning a failure?
    absl::Duration shutdown_timeout = absl::Seconds(60);
  };
  explicit DistributedRuntimeServiceImpl(const Options& options);
  ~DistributedRuntimeServiceImpl() override;

  DistributedRuntimeServiceImpl(const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl(DistributedRuntimeServiceImpl&&) = delete;
  DistributedRuntimeServiceImpl& operator=(
      const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl&& operator=(DistributedRuntimeServiceImpl&&) =
      delete;

  ::grpc::Status Connect(::grpc::ServerContext* context,
                         const ConnectRequest* request,
                         ConnectResponse* response) override;

  ::grpc::Status Shutdown(::grpc::ServerContext* context,
                          const ShutdownRequest* request,
                          ShutdownResponse* response) override;

  ::grpc::Status Heartbeat(::grpc::ServerContext* context,
                           const HeartbeatRequest* request,
                           HeartbeatResponse* response) override;

  ::grpc::Status EnumerateDevices(::grpc::ServerContext* context,
                                  const EnumerateDevicesRequest* request,
                                  EnumerateDevicesResponse* response) override;

  ::grpc::Status KeyValueGet(::grpc::ServerContext* context,
                             const KeyValueGetRequest* request,
                             KeyValueGetResponse* response) override;

  ::grpc::Status KeyValueSet(::grpc::ServerContext* context,
                             const KeyValueSetRequest* request,
                             KeyValueSetResponse* response) override;

 private:
  // Entry point for the heartbeat checking thread.
  void HeartbeatLoop();

  // Validates a session id number matches the current session id.
  xla::Status ValidateSessionId(uint64_t session_id);

  // Validates a node id number.
  xla::Status ValidateNodeId(int node_id);

  const Options options_;
  const uint64_t session_id_;

  absl::Mutex mu_;
  enum class State { kInitializing, kRunning, kClosed };
  State state_ ABSL_GUARDED_BY(mu_) = State::kInitializing;
  Status service_status_ ABSL_GUARDED_BY(mu_);

  // State for Connect() and heartbeats.
  struct Node {
    // Have we heard from a task with this ID?
    bool present = false;

    // A unique ID belonging to the client. Used to identify the client that
    // most recently called Connect() with a particular task id.
    uint64_t client_id = 0;

    // When did we last receive a heartbeat from this task?
    absl::Time last_heartbeat = absl::InfinitePast();
  };
  int num_nodes_present_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<Node> nodes_ ABSL_GUARDED_BY(mu_);

  // State for EnumerateDevices.
  int num_topologies_present_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<LocalTopologyProto> local_topologies_ ABSL_GUARDED_BY(mu_);
  absl::optional<GlobalTopologyProto> topology_ ABSL_GUARDED_BY(mu_);

  // State for Shutdown(). Counter of how many nodes are blocked at the
  // Shutdown() barrier.
  int num_nodes_shutting_down_ ABSL_GUARDED_BY(mu_) = 0;

  // Key-value store, used by distributed GPU code to share NCCL state.
  KeyValueStore key_value_store_;

  // Notification that tells the heartbeat thread to stop.
  absl::Notification stop_heartbeat_thread_;

  // Thread that checks for missing hearbeats from the clients periodically.
  std::unique_ptr<tensorflow::Thread> heartbeat_thread_;
};

class DistributedRuntimeService {
 public:
  static xla::StatusOr<std::unique_ptr<DistributedRuntimeService>> Get(
      const std::string& address,
      std::shared_ptr<::grpc::ServerCredentials> credentials,
      const DistributedRuntimeServiceImpl::Options& options);

  explicit DistributedRuntimeService(
      const DistributedRuntimeServiceImpl::Options& options);
  ~DistributedRuntimeService();

  DistributedRuntimeService(const DistributedRuntimeService&) = delete;
  DistributedRuntimeService(DistributedRuntimeService&&) = delete;
  DistributedRuntimeService& operator=(const DistributedRuntimeService&) =
      delete;
  DistributedRuntimeService& operator=(DistributedRuntimeService&&) = delete;

  void Shutdown();

  ::grpc::Server* server() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSserviceDTh mht_0(mht_0_v, 331, "", "./tensorflow/compiler/xla/pjrt/distributed/service.h", "server");
 return server_.get(); }

 private:
  DistributedRuntimeServiceImpl impl_;
  std::unique_ptr<::grpc::Server> server_;
};

// Everything below this point is exposed only for tests.

// Given a LocalTopologyProto object from each node, builds a
// GlobalTopologyProto that describes all nodes.
void BuildGlobalTopology(absl::Span<LocalTopologyProto> local_topologies,
                         GlobalTopologyProto* global_topology);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_
