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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTh() {
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


#include <functional>
#include <memory>
#include <string>

#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class DistributedRuntimeClient {
 public:
  struct Options {
    // This node's global ID. Required.
    int32_t node_id = -1;

    // Environment used for starting threads.
    tensorflow::Env* env = tensorflow::Env::Default();

    // RPC timeout used for RPC that don't have their own timeouts.
    absl::Duration rpc_timeout = absl::Seconds(120);

    // Time period for which Connect() should be retried. The client will keep
    // trying to open the initial connection for this period, even if any
    // individual Connect() RPC fails. May be zero, in which case Connect() will
    // only be attempted once.
    absl::Duration init_timeout = absl::ZeroDuration();

    // How long to wait for all nodes to call Shutdown(). If the timeout
    // expires, then shutdown() reports an error and returns control.
    absl::Duration shutdown_timeout = absl::Seconds(60);

    // Interval at which the client should send heartbeat RPCs to the
    // coordinator.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // How many failed heartbeat RPCs may fail due to a possibly-ephemeral
    // reason before we decide the coordinator has vanished and that we should
    // shut down.
    int max_missing_heartbeats = 10;

    // Callback invoked by the client when notification of a missing heartbeat
    // is reported by the coordinator, or we have not heard from the coordinator
    // recently. `coordinator_reported_failure` is true in the former case.
    // Exposed so tests can override this behavior to something non-fatal.
    std::function<void(xla::Status, bool coordinator_reported_failure)>
        missed_heartbeat_callback =
            [](xla::Status status, bool coordinator_reported_failure) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTh mht_0(mht_0_v, 238, "", "./tensorflow/compiler/xla/pjrt/distributed/client.h", "lambda");

              if (coordinator_reported_failure) {
                LOG(QFATAL)
                    << "Terminating process because the coordinator detected "
                       "missing heartbeats. This most likely indicates that "
                       "another task died; see the other task logs for more "
                       "details. Status: "
                    << status;
              } else {
                LOG(QFATAL)
                    << "Terminating process because of missing heartbeat "
                       "response from the coordinator. This most likely "
                       "indicates that the coordinator task died; see the "
                       "coordinator's task logs for more details. Status: "
                    << status;
              }
            };

    // For testing. Should the client explicitly Shutdown() on destruction?
    bool shutdown_on_destruction = true;
  };

  virtual ~DistributedRuntimeClient() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSdistributedPSclientDTh mht_1(mht_1_v, 263, "", "./tensorflow/compiler/xla/pjrt/distributed/client.h", "~DistributedRuntimeClient");
}

  // Connects to the master, and blocks until all clients have successfully
  // connected.
  // Not thread-safe, i.e., calls to Connect()/Shutdown()/EnumerateDevices()
  // must be serialized by some other means.
  virtual xla::Status Connect() = 0;

  // Reports to the master that the client is ready to shutdown, and blocks
  // until all clients are ready to shutdown or the shutdown timeout expires.
  // Not thread-safe.
  virtual xla::Status Shutdown() = 0;

  // Blocking enumeration of global devices. Used by the GPU platform.
  // Not thread-safe.
  virtual xla::Status EnumerateDevices(
      const LocalTopologyProto& local_topology,
      GlobalTopologyProto* global_topology) = 0;

  // The following APIs are thread-safe.
  virtual xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) = 0;

  virtual xla::Status KeyValueSet(std::string key, std::string value) = 0;
};

// Creates a distributed runtime client.
std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
