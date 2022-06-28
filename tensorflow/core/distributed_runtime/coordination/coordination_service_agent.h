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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTh() {
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
#include <string>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
class CoordinationServiceConfig;
class CoordinatedTask;
class Env;
class ServerDef;

// CoordinationServiceAgent defines the interface for tasks to communicate with
// the coordination service instance (which implements
// CoordinationServiceInterface). One instance of the agent should be deployed
// on each task for it to send various requests and stores / retrieves config
// key-value data to the service.
//
// See CoordinationServiceInterface for more details on coordination service.
//
// All coordination service errors will have an additional
// CoordinationServiceError payload to distinguish themselves from RPC failures.
// The payload can optionally specify the error origin, and if the error is
// reported by the user via `agent->ReportError()`.
//
// Possible service errors:
//    - errors::Internal: Coordination service is not enabled.
//                        If it was previously accessible, coordination service
//                        has been shut down.
//    - errors::Aborted: Incarnation mismatch during heartbeat (either remote
//                       task or coordination service has restarted).
//    - errors::Unavailable: Heartbeat timeout from remote task (failed,
//                           crashed or got preempted).
//    - errors::InvalidArgument: Unexpected heartbeat from remote task (not
//                               registered or wrong config).
class CoordinationServiceAgent {
 public:
  using StatusOrValueCallback =
      std::function<void(const StatusOr<std::string>&)>;
  using ChangedKeyValuesCallback =
      std::function<void(const std::map<std::string, std::string>&)>;

  virtual ~CoordinationServiceAgent() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTh mht_0(mht_0_v, 233, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h", "~CoordinationServiceAgent");
}

  // Initialize coordination service agent.
  virtual Status Initialize(
      Env* env, const ServerDef& server_def,
      std::unique_ptr<CoordinationClientCache> client_cache,
      StatusCallback error_fn) = 0;
  virtual Status Initialize(Env* env, const std::string& job_name, int task_id,
                            const CoordinationServiceConfig& configs,
                            std::unique_ptr<CoordinationClient> leader_client,
                            StatusCallback error_fn) = 0;
  virtual Status Initialize(Env* env, const CoordinatedTask& task,
                            const CoordinationServiceConfig& configs,
                            std::unique_ptr<CoordinationClient> leader_client,
                            StatusCallback error_fn) = 0;

  // Return true if the coordination service agent has been initialized.
  virtual bool IsInitialized() = 0;

  // Connect to coordination service with the following steps:
  //   - connect to service address specified in the config of `server_def`
  //   - register itself as a task to the service
  //   - start a thread to periodically send heartbeat message with the service
  // Possible service errors:
  //   - FailedPrecondition: Agent is not in DISCONNECTED state.
  //   - InvalidArgument: Unexpected task registration
  //   - Aborted: Duplicate task registration
  virtual Status Connect() = 0;

  // Wait for all tasks to be up and registered. The call blocks until all tasks
  // in the cluster are up, or some error occurs.
  // Possible service errors:
  //   - FailedPrecondition: Agent is not in RUNNING state.
  //   - InvalidArgument: Unexpected task request
  virtual Status WaitForAllTasks(
      const CoordinationServiceDeviceInfo& local_devices) = 0;

  // Get the device attributes of tasks from remote tasks in the cluster.
  virtual const CoordinationServiceDeviceInfo& GetClusterDeviceInfo() = 0;

  // State transition in coordination service agent:
  //
  //                 Init              Connect         SetError
  //   UNINITIALIZED ---> DISCONNECTED ------> RUNNING -------> ERROR
  //                           ^                                  |
  //                           |__________________________________|
  //                                         Reset
  enum class TaskState {
    UNINITIALIZED,
    DISCONNECTED,
    RUNNING,
    ERROR,
  };

  // Get status of a remote task.
  virtual StatusOr<TaskState> GetTaskStatus(const CoordinatedTask& task) = 0;

  // Report error to coordination service. This will invoke the error callback.
  // Note that the error payload will set `is_reported_error` to true, to
  // distinguish user-specified errors from internal service or RPC failures.
  // Possible service errors:
  //   - FailedPrecondition: Uninitialized/disconnected/already in error state.
  //   - InvalidArgument: Unexpected task request
  virtual Status ReportError(const Status& error) = 0;

  // Shuts down by disconnecting from the service. Should only be called if
  // agent is connected and no further agent calls (except the destructor) are
  // expected. If `shutdown_barrier_timeout_in_ms` is specified in the config,
  // blocks until all tasks reach the barrier before shutting down together. If
  // the barrier times out, this agent will still disconnect, while an error is
  // reported to other agents that did not reach the barrier on time.
  // Possible service errors:
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: Task was in error state (note: agent is still
  //                         shut down forcefully).
  virtual Status Shutdown() = 0;

  // Disconnect from the service, and clean up the internal error status.
  // Possible service errors:
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task is not in error state/has already
  //       disconnected.
  virtual Status Reset() = 0;

  // Get config key-value from the service.
  // Agent does not need to be connected to utilize the distributed key-value
  // store.
  //   - errors::DeadlineExceeded: timed out waiting for key.
  virtual StatusOr<std::string> GetKeyValue(const std::string& key) = 0;
  virtual StatusOr<std::string> GetKeyValue(const std::string& key,
                                            absl::Duration timeout) = 0;
  virtual void GetKeyValueAsync(const std::string& key,
                                StatusOrValueCallback done) = 0;

  // Insert config key-value to the service.
  //   - errors::AlreadyExists: key is already set.
  virtual Status InsertKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Delete config keys in the coordination service.
  virtual Status DeleteKeyValue(const std::string& key) = 0;

  // Update the value of a config key.
  virtual Status UpdateKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Register a callback that will be invoked when the key or keys under the key
  // directory are changed (inserted, deleted, or updated).
  virtual Status StartWatchKey(const std::string& key,
                               ChangedKeyValuesCallback on_change) = 0;
  virtual Status StopWatchKey(const std::string& key) = 0;

  // Blocks until all (or a subset of) tasks are at the barrier or the barrier
  // fails.
  //
  // `barrier_id` should be unique across barriers. Once the barrier has passed
  // or failed, subsequent calls will not block, and immediately respond with
  // the previous response.
  //
  // The first WaitAtBarrier() call received by the service for a particular
  // barrier_id is special in that it determines the barrier deadline based on
  // timeout duration.
  // However, if subsequent calls by different agents specify a different set of
  // `tasks` for the same `barrier_id`, the barrier will fail instantly.
  // For example,
  //   agent_1->WaitAtBarrier(“barrier”, 10min, <<”worker”, 1>, <”worker”, 2>>);
  //   agent_2->WaitAtBarrier(“barrier”, 10min, <<”worker”, 2>, <”worker”, 3>>);
  // Barrier fails after agent_2’s call because it specifies a different set of
  // participating tasks.
  //
  // If no tasks are specified (default), the barrier will block for all the
  // connected tasks.
  //
  // Possible service errors:
  //   - DeadlineExceeded: Timed out waiting for specified tasks at the barrier.
  //      Deadline is determined by the server timestamp when it receives the
  //      first WaitAtBarrier() + timeout duration.
  //   - Cancelled: One of the tasks called CancelBarrier().
  //   - Aborted: Service is shutting down.
  //   - Internal: Any participating task is in ERROR state.
  //   - InvalidArgument: (1) Conflicting tasks specified by different agents
  //       for the same barrier, (2) one of the participating tasks is not in
  //       the cluster, or (3) task making the request is not included in the
  //       list of participating tasks.
  //   - FailedPrecondition: Agent is in UNINITIALIZED or ERROR state.
  virtual Status WaitAtBarrier(const std::string& barrier_id,
                               absl::Duration timeout,
                               const std::vector<CoordinatedTask>& tasks) = 0;

  virtual void WaitAtBarrierAsync(const std::string& barrier_id,
                                  absl::Duration timeout,
                                  const std::vector<CoordinatedTask>& tasks,
                                  StatusCallback done) = 0;

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - FailedPrecondition: Barrier has already been passed.
  //   - NotFound: No barrier with the specified id is found.
  virtual Status CancelBarrier(const std::string& barrier_id) = 0;

 protected:
  // Set the service agent to error status and invoke the error callback.
  // Note: different from ReportError, this does not report the error status to
  // remote coordination service.
  virtual void SetError(const Status& error) = 0;

  // Activate the key-value callback watch.
  virtual Status ActivateWatch(const std::string& key,
                               const std::map<std::string, std::string>&) = 0;

 private:
  friend class CoordinationServiceRpcHandler;
};

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
