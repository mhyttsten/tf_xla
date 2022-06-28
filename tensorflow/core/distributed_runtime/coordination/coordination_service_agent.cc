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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc() {
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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

constexpr absl::Duration kDefaultClusterRegisterTimeout = absl::Hours(1);
constexpr absl::Duration kDefaultHeartbeatTimeout = absl::Seconds(10);
constexpr absl::Duration kDefaultShutdownTimeout = absl::Seconds(10);
constexpr char kHeartbeatThread[] = "CoordinationServiceHeartbeatLoop";

class CoordinationServiceAgentImpl : public CoordinationServiceAgent {
 public:
  CoordinationServiceAgentImpl() = default;
  ~CoordinationServiceAgentImpl() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "~CoordinationServiceAgentImpl");

    Status s = Shutdown();
    if (!s.ok()) {
      LOG(ERROR) << "Agent shutdown failed with status: " << s;
    }
  }
  Status Initialize(Env* env, const ServerDef& server_def,
                    std::unique_ptr<CoordinationClientCache> client_cache,
                    StatusCallback error_fn) override;
  Status Initialize(Env* env, const std::string& job_name, int task_id,
                    const CoordinationServiceConfig& configs,
                    std::unique_ptr<CoordinationClient> leader_client,
                    StatusCallback error_fn) override;
  Status Initialize(Env* env, const CoordinatedTask& task,
                    const CoordinationServiceConfig& configs,
                    std::unique_ptr<CoordinationClient> leader_client,
                    StatusCallback error_fn) override;
  bool IsInitialized() override;

  Status Connect() override;
  Status WaitForAllTasks(
      const CoordinationServiceDeviceInfo& local_devices) override;
  const CoordinationServiceDeviceInfo& GetClusterDeviceInfo() override;
  StatusOr<TaskState> GetTaskStatus(const CoordinatedTask& task) override;
  Status ReportError(const Status& error) override;
  Status Shutdown() override;
  Status Reset() override;

  StatusOr<std::string> GetKeyValue(const std::string& key) override;
  StatusOr<std::string> GetKeyValue(const std::string& key,
                                    absl::Duration timeout) override;
  void GetKeyValueAsync(const std::string& key,
                        StatusOrValueCallback done) override;
  Status InsertKeyValue(const std::string& key,
                        const std::string& value) override;
  Status DeleteKeyValue(const std::string& key) override;
  Status UpdateKeyValue(const std::string& key,
                        const std::string& value) override;

  Status StartWatchKey(const std::string& key,
                       ChangedKeyValuesCallback on_change) override;
  Status StopWatchKey(const std::string& key) override;
  Status WaitAtBarrier(const std::string& barrier_id, absl::Duration timeout,
                       const std::vector<CoordinatedTask>& tasks) override;
  void WaitAtBarrierAsync(const std::string& barrier_id, absl::Duration timeout,
                          const std::vector<CoordinatedTask>& tasks,
                          StatusCallback done) override;
  Status CancelBarrier(const std::string& barrier_id) override;

 protected:
  void SetError(const Status& error) override;
  Status ActivateWatch(const std::string& key,
                       const std::map<std::string, std::string>&) override;
  // Returns an error if agent is not running.
  Status ValidateRunningAgent();
  void StopHeartbeat();

 private:
  Env* env_ = nullptr;  // Not owned.
  const uint64_t incarnation_id_ = random::New64();
  CoordinatedTask task_;
  CoordinationServiceConfig configs_;
  std::unique_ptr<CoordinationClient> leader_client_;
  StatusCallback error_fn_;

  enum class State {
    UNINITIALIZED,
    DISCONNECTED,
    RUNNING,
    ERROR,
    SHUTDOWN,
  };
  mutable mutex state_mu_;
  State state_ TF_GUARDED_BY(state_mu_) = State::UNINITIALIZED;
  Status status_ TF_GUARDED_BY(state_mu_) = Status::OK();

  uint64_t leader_incarnation_ = 0;
  CoordinationServiceDeviceInfo cluster_devices_;

  mutex heartbeat_thread_shutdown_mu_;
  condition_variable heartbeat_thread_cv_;
  bool shutting_down_ TF_GUARDED_BY(heartbeat_thread_shutdown_mu_) = false;
  std::unique_ptr<Thread> heartbeat_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(CoordinationServiceAgentImpl);
};

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const ServerDef& server_def,
    std::unique_ptr<CoordinationClientCache> client_cache,
    StatusCallback error_fn) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_1(mht_1_v, 313, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Initialize");

  CoordinationServiceConfig configs =
      server_def.default_session_config().experimental().coordination_config();
  if (configs.service_leader().empty()) {
    const std::string& collective_leader = server_def.default_session_config()
                                               .experimental()
                                               .collective_group_leader();
    if (!collective_leader.empty()) {
      configs.set_service_leader(collective_leader);
      LOG(INFO) << "No coordination leader is set, using the collective leader "
                << collective_leader;
    } else {
      const std::string& default_leader =
          strings::StrCat("/job:", server_def.job_name(), "/replica:0/task:0");
      configs.set_service_leader(default_leader);
      LOG(INFO) << "No coordination leader is set, using the default leader "
                << default_leader;
    }
  }
  return Initialize(
      env, server_def.job_name(), server_def.task_index(), configs,
      client_cache->GetOwnedClient(configs.service_leader()), error_fn);
}

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const std::string& job_name, int task_id,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_2(mht_2_v, 345, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Initialize");

  CoordinatedTask task;
  task.set_job_name(job_name);
  task.set_task_id(task_id);
  return Initialize(env, task, configs, std::move(leader_client), error_fn);
}

Status CoordinationServiceAgentImpl::Initialize(
    Env* env, const CoordinatedTask& task,
    const CoordinationServiceConfig& configs,
    std::unique_ptr<CoordinationClient> leader_client,
    StatusCallback error_fn) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_3(mht_3_v, 359, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Initialize");

  mutex_lock l(state_mu_);
  if (state_ != State::UNINITIALIZED) {
    return MakeCoordinationError(errors::FailedPrecondition(
        "Coordination service agent has already been initialized."));
  }

  env_ = env;
  task_ = task;
  configs_ = configs;
  if (configs_.service_leader().empty()) {
    return MakeCoordinationError(errors::InvalidArgument(
        "CoordinationServiceAgent must be initialized with a valid leader."));
  }
  leader_client_ = std::move(leader_client);
  if (leader_client_ == nullptr) {
    return MakeCoordinationError(errors::InvalidArgument(
        "CoordinationServiceAgent must have a valid leader client."));
  }
  error_fn_ = error_fn;
  state_ = State::DISCONNECTED;
  return Status::OK();
}

bool CoordinationServiceAgentImpl::IsInitialized() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_4(mht_4_v, 386, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::IsInitialized");

  mutex_lock l(state_mu_);
  return state_ != State::UNINITIALIZED;
}

void CoordinationServiceAgentImpl::StopHeartbeat() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_5(mht_5_v, 394, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::StopHeartbeat");

  {
    mutex_lock l(heartbeat_thread_shutdown_mu_);
    shutting_down_ = true;
    heartbeat_thread_cv_.notify_all();
  }
  heartbeat_thread_.reset();
}

Status CoordinationServiceAgentImpl::Connect() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_6(mht_6_v, 406, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Connect");

  {
    mutex_lock l(state_mu_);
    if (state_ != State::DISCONNECTED) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent is not in DISCONNECTED state."));
    }
  }
  RegisterTaskRequest request;
  *request.mutable_source_task() = task_;
  request.set_incarnation(incarnation_id_);
  RegisterTaskResponse response;
  absl::Notification n;

  // Block until the remote service is up and the task is registered.
  CallOptions call_opts;
  const int64_t register_timeout =
      configs_.cluster_register_timeout_in_ms() > 0
          ? configs_.cluster_register_timeout_in_ms()
          : absl::ToInt64Milliseconds(kDefaultClusterRegisterTimeout);
  call_opts.SetTimeout(register_timeout);
  leader_client_->RegisterTaskAsync(
      &call_opts, &request, &response, [&](Status s) {
        if (!s.ok()) {
          SetError(s);
        } else {
          leader_incarnation_ = response.leader_incarnation();
          {
            mutex_lock l(state_mu_);
            state_ = State::RUNNING;
          }
        }
        n.Notify();
      });
  n.WaitForNotification();
  {
    mutex_lock l(state_mu_);
    if (state_ == State::ERROR) {
      return status_;
    }
  }

  heartbeat_thread_.reset(
      env_->StartThread(ThreadOptions(), kHeartbeatThread, [this]() -> void {
        HeartbeatRequest request;
        *request.mutable_source_task() = task_;
        request.set_incarnation(incarnation_id_);
        HeartbeatResponse response;
        const int64_t heartbeat_interval_ms =
            configs_.heartbeat_timeout_in_ms() > 0
                ? configs_.heartbeat_timeout_in_ms() / 2
                : absl::ToInt64Milliseconds(kDefaultHeartbeatTimeout) / 2;
        CallOptions call_opts;
        call_opts.SetTimeout(heartbeat_interval_ms);

        while (true) {
          {
            mutex_lock l(heartbeat_thread_shutdown_mu_);
            heartbeat_thread_cv_.wait_for(
                l, std::chrono::milliseconds(heartbeat_interval_ms));
            if (shutting_down_) {
              return;
            }
          }
          Status status;
          absl::Notification n;
          // Heartbeat RPC implementation automatically retries to tolerate
          // transient network failures.
          leader_client_->HeartbeatAsync(&call_opts, &request, &response,
                                         [&](Status s) {
                                           status = s;
                                           n.Notify();
                                         });
          n.WaitForNotification();
          if (!status.ok()) {
            SetError(status);
          } else if (response.leader_incarnation() != leader_incarnation_) {
            SetError(MakeCoordinationError(
                errors::Aborted("Leader incarnation ID mismatch: the "
                                "coordination leader has restarted.")));
          }
        }
      }));
  return Status::OK();
}

Status CoordinationServiceAgentImpl::WaitForAllTasks(
    const CoordinationServiceDeviceInfo& local_devices) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_7(mht_7_v, 496, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::WaitForAllTasks");

  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    return agent_running_status;
  }
  WaitForAllTasksRequest request;
  *request.mutable_source_task() = task_;
  *request.mutable_local_device_info() = local_devices;
  WaitForAllTasksResponse response;
  Status status;
  absl::Notification n;
  leader_client_->WaitForAllTasksAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  if (!status.ok()) {
    SetError(status);
    return status;
  }
  cluster_devices_.MergeFrom(response.cluster_device_info());
  return Status::OK();
}

const CoordinationServiceDeviceInfo&
CoordinationServiceAgentImpl::GetClusterDeviceInfo() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_8(mht_8_v, 524, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::GetClusterDeviceInfo");

  return cluster_devices_;
}

StatusOr<CoordinationServiceAgentImpl::TaskState>
CoordinationServiceAgentImpl::GetTaskStatus(const CoordinatedTask& task) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_9(mht_9_v, 532, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::GetTaskStatus");

  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgentImpl::GetTaskStatus is not implemented."));
}

Status CoordinationServiceAgentImpl::ReportError(const Status& error) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_10(mht_10_v, 540, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::ReportError");

  {
    mutex_lock l(state_mu_);
    if (state_ == State::UNINITIALIZED) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent must be initialized first before "
          "reporting error."));
    } else if (state_ == State::ERROR) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Coordination service agent is already in error state."));
    }
  }
  SetError(MakeCoordinationError(error, task_,
                                 /*is_reported_error=*/true));
  LOG(INFO) << "Reporting error to coordination service: " << error;
  ReportErrorToServiceRequest request;
  request.set_error_code(error.code());
  request.set_error_message(error.error_message());
  *request.mutable_error_origin() = task_;
  ReportErrorToServiceResponse response;

  absl::Notification n;
  leader_client_->ReportErrorToServiceAsync(&request, &response, [&](Status s) {
    if (!s.ok()) {
      LOG(ERROR) << "Encountered another error when reporting error to "
                    "coordination service: "
                 << s;
    }
    n.Notify();
  });
  n.WaitForNotification();
  return Status::OK();
}

Status CoordinationServiceAgentImpl::Shutdown() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_11(mht_11_v, 577, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Shutdown");

  Status status = Status::OK();
  bool is_connected = false;
  {
    mutex_lock l(state_mu_);
    is_connected = state_ == State::RUNNING;
  }
  // Disconnect agent from service.
  if (!configs_.agent_destruction_without_shutdown() && is_connected) {
    ShutdownTaskRequest request;
    *request.mutable_source_task() = task_;
    ShutdownTaskResponse response;
    CallOptions call_opts;
    const int64_t shutdown_timeout =
        configs_.shutdown_barrier_timeout_in_ms() > 0
            ? configs_.shutdown_barrier_timeout_in_ms()
            : absl::ToInt64Milliseconds(kDefaultShutdownTimeout);
    call_opts.SetTimeout(shutdown_timeout);

    absl::Notification n;
    leader_client_->ShutdownTaskAsync(&call_opts, &request, &response,
                                      [&status, &n](Status s) {
                                        status = s;
                                        n.Notify();
                                      });
    n.WaitForNotification();
    if (!status.ok()) {
      LOG(ERROR)
          << "Failed to disconnect from coordination service with status: "
          << status << ". Proceeding with agent shutdown anyway.";
    }
  }

  // Tear down agent.
  StopHeartbeat();
  {
    mutex_lock l(state_mu_);
    if (state_ == State::ERROR) {
      status = MakeCoordinationError(errors::FailedPrecondition(absl::StrCat(
          "Shutdown() was called while agent is in error state, implying that "
          "distributed execution failed. Note: agent will still shutdown "
          "anyway. Agent status: ",
          status_.ToString())));
    }
    state_ = State::SHUTDOWN;
  }
  return status;
}

Status CoordinationServiceAgentImpl::Reset() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_12(mht_12_v, 629, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::Reset");

  {
    mutex_lock l(state_mu_);
    if (state_ != State::ERROR) {
      return MakeCoordinationError(errors::FailedPrecondition(
          "Reset() failed: coordination service agent is not in ERROR state."));
    }
  }

  ResetTaskRequest request;
  *request.mutable_source_task() = task_;
  ResetTaskResponse response;

  Status status;
  absl::Notification n;
  leader_client_->ResetTaskAsync(&request, &response, [&status, &n](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  if (!status.ok()) {
    return status;
  }

  // Reset agent state.
  StopHeartbeat();
  {
    mutex_lock l(state_mu_);
    state_ = State::DISCONNECTED;
  }
  {
    mutex_lock l(heartbeat_thread_shutdown_mu_);
    shutting_down_ = false;
  }
  return status;
}

StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    const std::string& key) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_13(mht_13_v, 671, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::GetKeyValue");

  return GetKeyValue(key, /*timeout=*/absl::InfiniteDuration());
}

StatusOr<std::string> CoordinationServiceAgentImpl::GetKeyValue(
    const std::string& key, absl::Duration timeout) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_14(mht_14_v, 680, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::GetKeyValue");

  auto n = std::make_shared<absl::Notification>();
  auto result = std::make_shared<StatusOr<std::string>>();
  GetKeyValueAsync(key,
                   [n, result](const StatusOr<std::string>& status_or_value) {
                     *result = status_or_value;
                     n->Notify();
                   });
  bool call_completed_before_timeout =
      n->WaitForNotificationWithTimeout(timeout);
  if (!call_completed_before_timeout) {
    return MakeCoordinationError(errors::DeadlineExceeded(absl::Substitute(
        "GetKeyValue() timed out with key: $0 and duration: $1", key,
        absl::FormatDuration(timeout))));
  }
  return *result;
}

Status CoordinationServiceAgentImpl::InsertKeyValue(const std::string& key,
                                                    const std::string& value) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("key: \"" + key + "\"");
   mht_15_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_15(mht_15_v, 704, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::InsertKeyValue");

  InsertKeyValueRequest request;
  request.mutable_kv()->set_key(key.data(), key.size());
  request.mutable_kv()->set_value(value.data(), value.size());
  InsertKeyValueResponse response;

  Status status;
  absl::Notification n;
  leader_client_->InsertKeyValueAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::GetKeyValueAsync(
    const std::string& key, StatusOrValueCallback done) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_16(mht_16_v, 725, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::GetKeyValueAsync");

  auto request = std::make_shared<GetKeyValueRequest>();
  request->set_key(key);
  auto response = std::make_shared<GetKeyValueResponse>();
  leader_client_->GetKeyValueAsync(
      request.get(), response.get(),
      [request, response, done = std::move(done)](const Status& s) {
        if (!s.ok()) {
          done(s);
        } else {
          done(response->kv().value());
        }
      });
}

Status CoordinationServiceAgentImpl::DeleteKeyValue(const std::string& key) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_17(mht_17_v, 744, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::DeleteKeyValue");

  DeleteKeyValueRequest request;
  request.set_key(key);
  request.set_is_directory(true);
  DeleteKeyValueResponse response;

  Status status;
  absl::Notification n;
  leader_client_->DeleteKeyValueAsync(&request, &response, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return Status::OK();
}

Status CoordinationServiceAgentImpl::UpdateKeyValue(const std::string& key,
                                                    const std::string& value) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("key: \"" + key + "\"");
   mht_18_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_18(mht_18_v, 766, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::UpdateKeyValue");

  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::UpdateKeyValue is not implemented."));
}

Status CoordinationServiceAgentImpl::StartWatchKey(
    const std::string& key,
    CoordinationServiceAgentImpl::ChangedKeyValuesCallback on_change) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_19(mht_19_v, 777, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::StartWatchKey");

  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::StartWatchKey is not implemented."));
}

Status CoordinationServiceAgentImpl::StopWatchKey(const std::string& key) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_20(mht_20_v, 786, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::StopWatchKey");

  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::StopWatchKey is not implemented."));
}

void CoordinationServiceAgentImpl::SetError(const Status& error) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_21(mht_21_v, 794, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::SetError");

  assert(!error.ok());
  mutex_lock l(state_mu_);
  if (state_ == State::ERROR) return;
  state_ = State::ERROR;
  status_ = error;
  error_fn_(error);
}

Status CoordinationServiceAgentImpl::ActivateWatch(
    const std::string& key, const std::map<std::string, std::string>& kvs) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_22(mht_22_v, 808, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::ActivateWatch");

  return MakeCoordinationError(errors::Unimplemented(
      "CoordinationServiceAgent::ActivateWatch is not implemented."));
}

Status CoordinationServiceAgentImpl::WaitAtBarrier(
    const std::string& barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("barrier_id: \"" + barrier_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_23(mht_23_v, 819, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::WaitAtBarrier");

  Status status;
  absl::Notification n;
  WaitAtBarrierAsync(barrier_id, timeout, tasks, [&](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

void CoordinationServiceAgentImpl::WaitAtBarrierAsync(
    const std::string& barrier_id, absl::Duration timeout,
    const std::vector<CoordinatedTask>& tasks, StatusCallback done) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("barrier_id: \"" + barrier_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_24(mht_24_v, 836, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::WaitAtBarrierAsync");

  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    done(agent_running_status);
    return;
  }
  auto request = std::make_shared<BarrierRequest>();
  auto response = std::make_shared<BarrierResponse>();
  request->set_barrier_id(barrier_id);
  request->set_barrier_timeout_in_ms(timeout / absl::Milliseconds(1));
  *request->mutable_source_task() = task_;
  *request->mutable_tasks() = {tasks.begin(), tasks.end()};
  leader_client_->BarrierAsync(request.get(), response.get(),
                               [request, response, done = std::move(done)](
                                   const Status& s) { done(s); });
}

Status CoordinationServiceAgentImpl::CancelBarrier(
    const std::string& barrier_id) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("barrier_id: \"" + barrier_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_25(mht_25_v, 858, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::CancelBarrier");

  Status agent_running_status = ValidateRunningAgent();
  if (!agent_running_status.ok()) {
    return agent_running_status;
  }
  CancelBarrierRequest request;
  CancelBarrierResponse response;
  request.set_barrier_id(barrier_id);
  *request.mutable_source_task() = task_;

  Status status;
  absl::Notification n;
  leader_client_->CancelBarrierAsync(&request, &response, [&](const Status& s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  return status;
}

// Returns an error if agent is not running.
Status CoordinationServiceAgentImpl::ValidateRunningAgent() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agentDTcc mht_26(mht_26_v, 882, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent.cc", "CoordinationServiceAgentImpl::ValidateRunningAgent");

  mutex_lock l(state_mu_);
  switch (state_) {
    case State::RUNNING:
      return Status::OK();

    case State::UNINITIALIZED:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently UNINITIALIZED."));

    case State::DISCONNECTED:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently DISCONNECTED."));

    case State::ERROR:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently in ERROR."));

    case State::SHUTDOWN:
      return MakeCoordinationError(errors::FailedPrecondition(
          "Agent must be in RUNNING state. It is currently in SHUTDOWN."));

    default:
      return MakeCoordinationError(errors::FailedPrecondition(absl::StrCat(
          "Agent is not in RUNNING state. Current state: ", state_)));
  }
}

}  // namespace

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent() {
  return std::make_unique<CoordinationServiceAgentImpl>();
}

}  // namespace tensorflow
