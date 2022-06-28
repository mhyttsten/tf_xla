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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/worker.h"

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/error_payloads.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/device_profiler_session.h"
#include "tensorflow/core/protobuf/distributed_runtime_payloads.pb.h"

namespace tensorflow {

Worker::Worker(WorkerEnv* env) : env_(env), recent_request_ids_(100000) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::Worker");

  // Enable log history collection in StatusGroup so that recent warning and
  // error log messages will be attached to the root error status to be
  // forwarded to the master.
  StatusGroup::ConfigureLogHistory();
}

void Worker::GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                            GetStatusResponse* response, bool fail_fast,
                            StatusCallback done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::GetStatusAsync");

  const DeviceMgr* dm = env_->device_mgr;
  std::vector<DeviceAttributes> devices;
  dm->ListDeviceAttributes(&devices);
  response->mutable_device_attributes()->Reserve(devices.size());
  for (auto& d : devices) {
    response->add_device_attributes()->Swap(&d);
  }
  done(Status::OK());
}

void Worker::CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                      CreateWorkerSessionResponse* response,
                                      StatusCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CreateWorkerSessionAsync");

  Status s = env_->session_mgr->CreateSession(
      request->session_handle(), request->server_def(),
      request->cluster_device_attributes(), request->isolate_session_state(),
      request->master_task(), request->master_incarnation());
  done(s);
}

void Worker::DeleteWorkerSessionAsync(CallOptions* opts,
                                      const DeleteWorkerSessionRequest* request,
                                      DeleteWorkerSessionResponse* response,
                                      StatusCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::DeleteWorkerSessionAsync");

  Status s = env_->session_mgr->DeleteSession(request->session_handle());
  done(s);
}

void Worker::RegisterGraphAsync(const RegisterGraphRequest* request,
                                RegisterGraphResponse* response,
                                StatusCallback done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::RegisterGraphAsync");

  std::shared_ptr<WorkerSession> session;
  Status s;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (s.ok()) {
    s = session->graph_mgr()->Register(
        request->session_handle(), request->graph_def(),
        request->graph_options(), request->debug_options(),
        request->config_proto(), request->collective_graph_key(), session.get(),
        session->cluster_flr(), response->mutable_graph_handle());
  }
  done(s);
}

void Worker::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                  DeregisterGraphResponse* response,
                                  StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_5(mht_5_v, 279, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::DeregisterGraphAsync");

  std::shared_ptr<WorkerSession> session;
  Status s;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (s.ok()) {
    s = session->graph_mgr()->Deregister(request->graph_handle());
  }

  done(s);
}

void Worker::AbortStep(int64_t step_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::AbortStep");

  RemoteRendezvous* rendez = env_->rendezvous_mgr->Find(step_id);
  // Do not abort if it's a context global instance for eager op-by-op execution
  if (rendez->IsRemoteEagerContextDefault()) return;
  SchedNonBlockingClosureAfter(1000000, [rendez, step_id]() {
    // Delay a bit before aborting the step. This way, the root
    // cause may return first back to the client instead of this
    // cancellation generated abort error.
    rendez->StartAbort(errors::Aborted("Step ", step_id,
                                       " cancelled.  Cancelling rendezvous."));
    rendez->Unref();
  });
}

Status Worker::PrepareRunGraph(RunGraphRequestWrapper* req,
                               GraphMgr::NamedTensors* in,
                               GraphMgr::NamedTensors* out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_7(mht_7_v, 317, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::PrepareRunGraph");

  static Tensor empty_tensor(DT_FLOAT);
  if (req->num_sends() > 0) {
    Tensor val;
    for (size_t i = 0; i < req->num_sends(); ++i) {
      TF_RETURN_IF_ERROR(req->SendValue(i, &val));
      in->insert({req->send_key(i), val});
    }
  }
  for (size_t i = 0; i < req->num_recvs(); ++i) {
    out->insert({req->recv_key(i), empty_tensor});
  }
  return Status::OK();
}

void Worker::RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                           MutableRunGraphResponseWrapper* response,
                           StatusCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_8(mht_8_v, 337, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::RunGraphAsync");

  if (request->store_errors_in_response_body()) {
    done = [response, done](const Status& status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_9(mht_9_v, 342, "", "./tensorflow/core/distributed_runtime/worker.cc", "lambda");

      response->set_status(status);
      done(Status::OK());
    };
  }
  if (request->is_partial()) {
    DoPartialRunGraph(opts, request, response, std::move(done));
  } else {
    DoRunGraph(opts, request, response, std::move(done));
  }
}

MutableRunGraphRequestWrapper* Worker::CreateRunGraphRequest() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_10(mht_10_v, 357, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CreateRunGraphRequest");

  return new InMemoryRunGraphRequest;
}

MutableRunGraphResponseWrapper* Worker::CreateRunGraphResponse() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_11(mht_11_v, 364, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CreateRunGraphResponse");

  return new InMemoryRunGraphResponse;
}

void Worker::DoRunGraph(CallOptions* opts, RunGraphRequestWrapper* request,
                        MutableRunGraphResponseWrapper* response,
                        StatusCallback done) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_12(mht_12_v, 373, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::DoRunGraph");

  const int64_t step_id = request->step_id();
  TRACEPRINTF("RunGraph: %lld", step_id);
  Status s = recent_request_ids_.TrackUnique(request->request_id(),
                                             "RunGraph (Worker)", request);
  if (!s.ok()) {
    done(s);
    return;
  }

  std::shared_ptr<WorkerSession> session;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (!s.ok()) {
    done(s);
    return;
  }
  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  s = PrepareRunGraph(request, &in, out);
  if (!s.ok()) {
    delete out;
    done(s);
    return;
  }
  StepStatsCollector* collector = nullptr;
  if (request->exec_opts().report_tensor_allocations_upon_oom() ||
      request->exec_opts().record_timeline() ||
      request->exec_opts().record_costs()) {
    collector = new StepStatsCollector(response->mutable_step_stats());
  }
  DeviceProfilerSession* device_profiler_session = nullptr;
  if (collector && request->exec_opts().record_timeline()) {
    // If timeline was requested, assume we want hardware level tracing.
    device_profiler_session = DeviceProfilerSession::Create().release();
  }
  CancellationManager* cm = new CancellationManager;
  opts->SetCancelCallback([this, cm, step_id]() {
    LOG(INFO) << "Cancellation requested for RunGraph.";
    cm->StartCancel();
    AbortStep(step_id);
  });
  CancellationToken token;
  token = cancellation_manager_.get_cancellation_token();
  bool already_cancelled = !cancellation_manager_.RegisterCallback(
      token, [cm]() { cm->StartCancel(); });
  if (already_cancelled) {
    opts->ClearCancelCallback();
    delete cm;
    delete collector;
    delete device_profiler_session;
    delete out;
    done(errors::Aborted("Call was aborted"));
    return;
  }
  session->graph_mgr()->ExecuteAsync(
      request->graph_handle(), step_id, request->exec_opts(), in, session.get(),
      collector, response, cm, env_->session_mgr->GetCoordinationServiceAgent(),
      [this, step_id, response, session, cm, out, token, collector,
       device_profiler_session, opts, done](const Status& status) {
        Status s = status;
        if (s.ok()) {
          s = session->graph_mgr()->RecvOutputs(step_id, out);
        }

        opts->ClearCancelCallback();
        cancellation_manager_.DeregisterCallback(token);
        delete cm;

        if (device_profiler_session) {
          device_profiler_session->CollectData(response->mutable_step_stats())
              .IgnoreError();
        }

        if (s.ok()) {
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }

        if (collector) collector->Finalize();
        delete collector;
        delete device_profiler_session;
        delete out;
        done(s);
      });
}

// TODO(suharshs): Add stats collection support to partial run.
void Worker::DoPartialRunGraph(CallOptions* opts,
                               RunGraphRequestWrapper* request,
                               MutableRunGraphResponseWrapper* response,
                               StatusCallback done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_13(mht_13_v, 474, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::DoPartialRunGraph");

  const int64_t step_id = request->step_id();
  const string& graph_handle = request->graph_handle();
  TRACEPRINTF("PartialRunGraph: %lld", step_id);
  Status s = recent_request_ids_.TrackUnique(
      request->request_id(), "PartialRunGraph (Worker)", request);
  if (!s.ok()) {
    done(s);
    return;
  }

  std::shared_ptr<WorkerSession> session;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  s = PrepareRunGraph(request, &in, out);
  auto finish = [done, out, opts](const Status& s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_14(mht_14_v, 503, "", "./tensorflow/core/distributed_runtime/worker.cc", "lambda");

    opts->ClearCancelCallback();
    delete out;
    done(s);
  };
  if (!s.ok()) {
    finish(s);
    return;
  }

  CancellationManager* cm = nullptr;
  bool is_new_partial_run = partial_run_mgr_.FindOrCreate(step_id, &cm);

  // Before we start doing anything, we set the RPC cancellation.
  opts->SetCancelCallback([this, cm, step_id]() {
    LOG(INFO) << "Cancellation requested for PartialRunGraph.";
    cm->StartCancel();
    AbortStep(step_id);
  });

  // If this is a new partial run request, the request will need to start the
  // executors.
  if (is_new_partial_run) {
    CancellationToken token;
    token = cancellation_manager_.get_cancellation_token();
    cancellation_manager_.RegisterCallback(token,
                                           [cm]() { cm->StartCancel(); });
    session->graph_mgr()->ExecuteAsync(
        graph_handle, step_id, request->exec_opts(), in, session.get(),
        /*collector=*/nullptr, /*response=*/nullptr, cm,
        env_->session_mgr->GetCoordinationServiceAgent(),
        [this, token, step_id, session](Status s) {
          cancellation_manager_.DeregisterCallback(token);
          partial_run_mgr_.ExecutorDone(step_id, s);
        });
  } else {
    // Send the partial run's new inputs.
    s = session->graph_mgr()->SendInputs(step_id, in);
    if (!s.ok()) {
      finish(s);
      return;
    }
  }

  session->graph_mgr()->RecvOutputsAsync(
      step_id, out, [this, out, request, response, step_id, finish](Status s) {
        if (s.ok()) {
          // Construct and return the resp.
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }
        if (request->is_last_partial_run()) {
          partial_run_mgr_.PartialRunDone(step_id, finish, s);
        } else {
          finish(s);
        }
      });
}

void Worker::CleanupGraphAsync(const CleanupGraphRequest* request,
                               CleanupGraphResponse* response,
                               StatusCallback done) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_15(mht_15_v, 570, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CleanupGraphAsync");

  const int64_t step_id = request->step_id();
  env_->rendezvous_mgr->Cleanup(step_id);
  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->Cleanup(step_id);
  }
  for (Device* d : env_->local_devices) {
    ScopedAllocatorMgr* sam = d->GetScopedAllocatorMgr();
    if (sam) {
      sam->Cleanup(step_id);
    }
  }
  done(Status::OK());
}

void Worker::CleanupAllAsync(const CleanupAllRequest* request,
                             CleanupAllResponse* response,
                             StatusCallback done) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_16(mht_16_v, 590, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CleanupAllAsync");

  std::vector<string> containers;
  for (const auto& c : request->container()) containers.push_back(c);
  env_->device_mgr->ClearContainers(containers);
  done(Status::OK());
}

void Worker::LoggingAsync(const LoggingRequest* request,
                          LoggingResponse* response, StatusCallback done) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_17(mht_17_v, 601, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::LoggingAsync");

  done(errors::Unimplemented("Logging"));
}

void Worker::TracingAsync(const TracingRequest* request,
                          TracingResponse* response, StatusCallback done) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_18(mht_18_v, 609, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::TracingAsync");

  done(errors::Unimplemented("Tracing"));
}

void Worker::RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                          RecvBufResponse* response, StatusCallback done) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_19(mht_19_v, 617, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::RecvBufAsync");

  // The base Worker class does not implement RecvBufAsync because
  // it is not currently used for worker-to-worker communication. Use a
  // transport-specific implementation (such as `GrpcWorker::RecvBufAsync()`)
  // instead.
  done(errors::Unimplemented("Worker::RecvBufAsync()"));
}

void Worker::CompleteGroupAsync(CallOptions* opts,
                                const CompleteGroupRequest* request,
                                CompleteGroupResponse* response,
                                StatusCallback done) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_20(mht_20_v, 631, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CompleteGroupAsync");

  if (!request->has_device_attributes()) {
    done(errors::Internal(
        "CompleteGroupRequest device_attributes is not set. Make sure you're "
        "running the same version of Tensorflow on all workers."));
    return;
  }
  if (env_->collective_executor_mgr) {
    auto group_params = new CollGroupParams();
    group_params->group_key = request->group_key();
    group_params->group_size = request->group_size();
    group_params->device_type = DeviceType(request->device_type());
    env_->collective_executor_mgr->GetParamResolver()->CompleteGroupAsync(
        request->device_attributes(), group_params, &cancellation_manager_,
        [response, group_params, done = std::move(done)](const Status& s) {
          if (s.ok()) {
            response->set_group_key(group_params->group_key);
            response->set_group_size(group_params->group_size);
            response->set_device_type(group_params->device_type.type_string());
            response->set_num_tasks(group_params->num_tasks);
            for (const CollGroupMember& member : group_params->members) {
              *response->add_device_attributes() = member.device;
            }
            response->set_communicator_key(
                group_params->runtime_details.communicator_key);
          } else {
            LOG(ERROR) << "Bad status from CompleteGroupDistributed: " << s;
          }
          delete group_params;
          done(s);
        });
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

void Worker::CompleteInstanceAsync(CallOptions* opts,
                                   const CompleteInstanceRequest* request,
                                   CompleteInstanceResponse* response,
                                   StatusCallback done) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_21(mht_21_v, 674, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::CompleteInstanceAsync");

  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->GetParamResolver()->CompleteInstanceAsync(
        request, response, &cancellation_manager_, done);
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

void Worker::GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                  GetStepSequenceResponse* response,
                                  StatusCallback done) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_22(mht_22_v, 689, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::GetStepSequenceAsync");

  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->GetStepSequenceAsync(request, response,
                                                        done);
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

// Helper for RecvTensor. Validates "key" and returns the source
// device in "*src_dev".
Status Worker::PrepareRecvTensor(const Rendezvous::ParsedKey& parsed,
                                 Device** src_dev) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_23(mht_23_v, 705, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::PrepareRecvTensor");

  // Figures out which device the tensor is hosted on.
  string local_name = DeviceNameUtils::LocalName(parsed.src_device);
  TF_RETURN_IF_ERROR(env_->device_mgr->LookupDevice(local_name, src_dev));

  // Does the device have the right incarnation number we expect?
  if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
    return errors::AbortedWithPayloads(
        strings::StrCat("RecvTensor expects a different device incarnation: ",
                        parsed.src_incarnation, " vs. ",
                        (*src_dev)->attributes().incarnation(),
                        ". Your worker job (\"",
                        env_->session_mgr->LegacySession()->worker_name(),
                        "\") was probably restarted. Check your "
                        "worker job for the reason why it was restarted."),
        {{kWorkerPossiblyRestarted,
          distributed_runtime::WorkerPossiblyRestarted().SerializeAsString()}});
  }

  return Status::OK();
}

void Worker::RecvTensorAsync(CallOptions* opts,
                             const RecvTensorRequest* request,
                             TensorResponse* response, StatusCallback done) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworkerDTcc mht_24(mht_24_v, 732, "", "./tensorflow/core/distributed_runtime/worker.cc", "Worker::RecvTensorAsync");

  // The base Worker class does not implement RecvTensorAsync, because
  // it is not currently used for worker-to-worker communication. Use a
  // transport-specific implementation (such as `GrpcWorker::RecvTensorAsync()`)
  // instead.
  done(errors::Unimplemented("Worker::RecvTensorAsync()"));
}

}  // namespace tensorflow
