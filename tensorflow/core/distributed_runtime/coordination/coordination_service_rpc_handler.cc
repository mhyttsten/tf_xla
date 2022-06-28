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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc() {
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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.h"

#include <string>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {

void CoordinationServiceRpcHandler::SetAgentInstance(
    CoordinationServiceAgent* agent) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::SetAgentInstance");

  mutex_lock l(agent_mu_);
  agent_ = agent;
}

void CoordinationServiceRpcHandler::RegisterTaskAsync(
    const RegisterTaskRequest* request, RegisterTaskResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::RegisterTaskAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service->GetServiceIncarnation();
  response->set_leader_incarnation(leader_incarnation);
  done(service->RegisterTask(task, incarnation));
}

void CoordinationServiceRpcHandler::HeartbeatAsync(
    const HeartbeatRequest* request, HeartbeatResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::HeartbeatAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  const CoordinatedTask& task = request->source_task();
  const uint64_t incarnation = request->incarnation();
  const uint64_t leader_incarnation = service->GetServiceIncarnation();
  Status s = service->RecordHeartbeat(task, incarnation);
  if (!s.ok()) {
    done(s);
    return;
  }
  response->set_leader_incarnation(leader_incarnation);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::WaitForAllTasksAsync(
    const WaitForAllTasksRequest* request, WaitForAllTasksResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::WaitForAllTasksAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service->WaitForAllTasks(
      request->source_task(), request->local_device_info(),
      [response, service, done = std::move(done)](Status s) {
        if (s.ok()) {
          *response->mutable_cluster_device_info() =
              service->ListClusterDevices();
        }
        done(s);
      });
}

void CoordinationServiceRpcHandler::ShutdownTaskAsync(
    const ShutdownTaskRequest* request, ShutdownTaskResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::ShutdownTaskAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  service->ShutdownTaskAsync(request->source_task(),
                             [done](Status s) { done(s); });
}

void CoordinationServiceRpcHandler::ResetTaskAsync(
    const ResetTaskRequest* request, ResetTaskResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_5(mht_5_v, 298, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::ResetTaskAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->ResetTask(request->source_task()));
}

void CoordinationServiceRpcHandler::ReportErrorToTaskAsync(
    const ReportErrorToTaskRequest* request,
    ReportErrorToTaskResponse* response, StatusCallback done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_6(mht_6_v, 314, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::ReportErrorToTaskAsync");

  tf_shared_lock l(agent_mu_);
  if (agent_ == nullptr) {
    done(MakeCoordinationError(errors::Internal(
        "CoordinationServiceAgent is uninitialized or has already shutdown.")));
    return;
  }
  const CoordinationServiceError& error_payload = request->error_payload();
  Status error(static_cast<error::Code>(request->error_code()),
               strings::StrCat("Error reported from /job:",
                               error_payload.source_task().job_name(),
                               "/task:", error_payload.source_task().task_id(),
                               ": ", request->error_message()));
  error = MakeCoordinationError(error, error_payload);
  agent_->SetError(error);
  done(Status::OK());
}

void CoordinationServiceRpcHandler::ReportErrorToServiceAsync(
    const ReportErrorToServiceRequest* request,
    ReportErrorToServiceResponse* response, StatusCallback done) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_7(mht_7_v, 337, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::ReportErrorToServiceAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->ReportTaskError(
      request->error_origin(),
      MakeCoordinationError(
          Status{static_cast<error::Code>(request->error_code()),
                 request->error_message()},
          request->error_origin(),
          /*is_reported_error=*/true)));
}

void CoordinationServiceRpcHandler::InsertKeyValueAsync(
    const InsertKeyValueRequest* request, InsertKeyValueResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_8(mht_8_v, 359, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::InsertKeyValueAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->InsertKeyValue(request->kv().key(), request->kv().value()));
}

void CoordinationServiceRpcHandler::GetKeyValueAsync(
    const GetKeyValueRequest* request, GetKeyValueResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_9(mht_9_v, 375, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::GetKeyValueAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  response->mutable_kv()->set_key(request->key());
  service->GetKeyValueAsync(
      request->key(), [response, done = std::move(done)](
                          const StatusOr<std::string>& status_or_value) {
        if (status_or_value.ok()) {
          response->mutable_kv()->set_value(status_or_value.ValueOrDie());
        }
        done(status_or_value.status());
      });
}

void CoordinationServiceRpcHandler::DeleteKeyValueAsync(
    const DeleteKeyValueRequest* request, DeleteKeyValueResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_10(mht_10_v, 399, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::DeleteKeyValueAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->DeleteKeyValue(request->key()));
}

void CoordinationServiceRpcHandler::BarrierAsync(const BarrierRequest* request,
                                                 BarrierResponse* response,
                                                 StatusCallback done) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_11(mht_11_v, 415, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::BarrierAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  std::vector<CoordinatedTask> tasks = {request->tasks().begin(),
                                        request->tasks().end()};
  service->BarrierAsync(
      request->barrier_id(),
      absl::Milliseconds(request->barrier_timeout_in_ms()),
      request->source_task(), tasks,
      [done = std::move(done)](const Status& status) { done(status); });
}

void CoordinationServiceRpcHandler::CancelBarrierAsync(
    const CancelBarrierRequest* request, CancelBarrierResponse* response,
    StatusCallback done) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_rpc_handlerDTcc mht_12(mht_12_v, 437, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_rpc_handler.cc", "CoordinationServiceRpcHandler::CancelBarrierAsync");

  CoordinationServiceInterface* service =
      CoordinationServiceInterface::GetCoordinationServiceInstance();
  if (service == nullptr) {
    done(MakeCoordinationError(
        errors::Internal("Coordination service is not enabled.")));
    return;
  }
  done(service->CancelBarrier(request->barrier_id(), request->source_task()));
}

}  // namespace tensorflow
