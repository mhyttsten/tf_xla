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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh() {
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
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
class CoordinationServiceDeviceInfo;
class ServerDef;
class Env;

// Static registration for coordination service implementations.
#define REGISTER_COORDINATION_SERVICE(service_type_name, factory_fn)        \
  REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(__COUNTER__, service_type_name, \
                                            factory_fn)
#define REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(counter, service_type_name, \
                                                  factory_fn)                 \
  static bool static_coordination_service_##counter TF_ATTRIBUTE_UNUSED =     \
      []() {                                                                  \
        ::tensorflow::CoordinationServiceInterface::                          \
            RegisterCoordinationService(service_type_name,                    \
                                        std::move(factory_fn));               \
        return true;                                                          \
      }()

// Coordination service is used for controlling and coordinating distributed
// execution in a cluster of multiple tasks.
//
// When enabled, the service keeps track of cluster configurations and the state
// of cluster members. TF runtime and libraries can use it to orchastrate
// cluster initialization, check the healthiness of tasks, and propagate error
// messages to the cluster.
//
// Normally, the service should first Start(), then perform the supported
// coordination operations, and finally Stop(). When service runs into error or
// SetError() is called, all subsequent operations will be in error state.
//
// CoordinationServiceInterface defines the service interface for distributed
// coordination. One instance of the service should be deployed in a cluster,
// handling various requests and stores configuration key-value data for the
// tasks. Each task interacts with the service through CoordinationServiceAgent.
class CoordinationServiceInterface {
 public:
  using CoordinationServiceFactory =
      std::function<std::unique_ptr<CoordinationServiceInterface>(
          Env* env, const ServerDef& server_def,
          std::unique_ptr<CoordinationClientCache> cache)>;

  using StatusOrValueCallback =
      std::function<void(const StatusOr<std::string>&)>;

  virtual ~CoordinationServiceInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh mht_0(mht_0_v, 243, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service.h", "~CoordinationServiceInterface");
}

  static void RegisterCoordinationService(
      const std::string& service_type_name,
      CoordinationServiceFactory factory_fn) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("service_type_name: \"" + service_type_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh mht_1(mht_1_v, 251, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service.h", "RegisterCoordinationService");

    auto factories = GetCoordinationServiceFactories();
    factories->emplace(service_type_name, factory_fn);
  }

  static std::unique_ptr<CoordinationServiceInterface>
  EnableCoordinationService(const std::string& service_type, Env* env,
                            const ServerDef& server_def,
                            std::unique_ptr<CoordinationClientCache> cache) {
    const auto* factories = GetCoordinationServiceFactories();
    auto factories_iter = factories->find(service_type);
    if (factories_iter == factories->end()) {
      LOG(ERROR) << "No coordination service factory found for service type "
                 << service_type;
      return nullptr;
    }
    auto service = factories_iter->second(env, server_def, std::move(cache));
    if (service != nullptr) {
      *GetCoordinationServiceInstancePtr() = service.get();
    }
    return service;
  }

  static CoordinationServiceInterface* GetCoordinationServiceInstance() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh mht_2(mht_2_v, 277, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service.h", "GetCoordinationServiceInstance");

    return *GetCoordinationServiceInstancePtr();
  }

  // Register a task to the service.
  virtual Status RegisterTask(const CoordinatedTask& task,
                              uint64_t incarnation) = 0;

  // Wait for all tasks to be up and running, and register local device
  // info. The callback is invoked when all tasks are up and registered, or some
  // error occurs.
  virtual void WaitForAllTasks(const CoordinatedTask& task,
                               const CoordinationServiceDeviceInfo& devices,
                               StatusCallback done) = 0;

  // Disconnects task from the service. If `shutdown_barrier_timeout_in_ms` is
  // specified in the config, blocks until all tasks reach the barrier before
  // disconnecting together.
  // Possible service errors:
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  virtual void ShutdownTaskAsync(const CoordinatedTask& task,
                                 StatusCallback done) = 0;

  // Disconnects task from the service and cleans up its internal error state.
  // Possible service errors:
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  virtual Status ResetTask(const CoordinatedTask& task) = 0;

  // Update the heartbeat timestamp of a task. This should only be invoked on
  // the leader of the cluster.
  virtual Status RecordHeartbeat(const CoordinatedTask& task,
                                 uint64_t incarnation) = 0;

  // Set a task in error state permanently.
  virtual Status ReportTaskError(const CoordinatedTask& task, Status error) = 0;

  // Insert a configuration key-value in the coordination service.
  // For now, a key-value can only be inserted once and cannot be updated.
  // The key-values are not persisted and will be lost if the leader fails.
  virtual Status InsertKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Get a configuration key-value from the coordination service. Block until
  // the key-value is available.
  virtual StatusOr<std::string> GetKeyValue(const std::string& key) = 0;
  // Get a configuration key-value from the coordination service. The `done`
  // callback is invoked when the key-value becomes available.
  virtual void GetKeyValueAsync(const std::string& key,
                                StatusOrValueCallback done) = 0;

  // Delete configuration key-value. If key is a directory, recursively clean
  // up all key-values under the directory.
  virtual Status DeleteKeyValue(const std::string& key) = 0;

  // Blocks until all (or a subset of) tasks are at the barrier or the barrier
  // fails.
  //
  // `barrier_id` should be unique across barriers. Once the barrier has passed
  // or failed, subsequent calls will not block, and immediately respond with
  // the previous response.
  //
  // The first WaitAtBarrier() call received by the service for a particular
  // barrier id is special in that it determines the barrier deadline based on
  // timeout duration.
  // However, if subsequent calls by different agents specify a different set of
  // `participating_tasks` for the same `barrier_id`, the barrier will fail
  // instantly.
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
  virtual void BarrierAsync(
      const std::string& barrier_id, absl::Duration timeout,
      const CoordinatedTask& task,
      const std::vector<CoordinatedTask>& participating_tasks,
      StatusCallback done) = 0;

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - FailedPrecondition: Barrier has already been passed.
  //   - NotFound: No barrier with the specified id is found.
  virtual Status CancelBarrier(const std::string& barrier_id,
                               const CoordinatedTask& task) = 0;

 protected:
  // TODO(haoyuzhang): Remove singleton once we decide on how to access the
  // coordination service from op kernel.
  static CoordinationServiceInterface** GetCoordinationServiceInstancePtr() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_serviceDTh mht_3(mht_3_v, 383, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service.h", "GetCoordinationServiceInstancePtr");

    static CoordinationServiceInterface* instance = nullptr;
    return &instance;
  }

 private:
  friend class CoordinationServiceRpcHandler;
  friend class CoordinationServiceTest_ListClusterDevices_TfDevice_Test;
  friend class CoordinationServiceTest_ListClusterDevices_XlaDevice_Test;

  virtual const CoordinationServiceDeviceInfo& ListClusterDevices() = 0;
  virtual uint64_t GetServiceIncarnation() = 0;

  static std::unordered_map<std::string, CoordinationServiceFactory>*
  GetCoordinationServiceFactories() {
    static auto* coordination_service_factories =
        new std::unordered_map<std::string, CoordinationServiceFactory>();
    return coordination_service_factories;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
