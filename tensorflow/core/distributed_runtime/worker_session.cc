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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc() {
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
#include "tensorflow/core/distributed_runtime/worker_session.h"

#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/gauge.h"

namespace tensorflow {

namespace {

auto* worker_session_created =
    monitoring::Gauge<bool, 0>::New("/tensorflow/core/worker_session_created",
                                    "True if a worker session was created.");

// A private cache that wraps worker_cache and allows reuse of
// WorkerInterface objects.
class WorkerFreeListCache : public WorkerCacheInterface {
 public:
  explicit WorkerFreeListCache(std::unique_ptr<WorkerCacheInterface> w)
      : wrapped_(std::move(w)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerFreeListCache");
}

  ~WorkerFreeListCache() final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "~WorkerFreeListCache");

    for (auto& p : workers_) {
      wrapped_->ReleaseWorker(p.first, p.second.worker);
    }
  }

  void ListWorkers(std::vector<string>* workers) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "ListWorkers");

    wrapped_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "ListWorkersInJob");

    wrapped_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "GetOrCreateWorker");

    mutex_lock l(mu_);
    auto p = workers_.find(target);
    if (p != workers_.end()) {
      return p->second.worker;
    }
    WorkerState state;
    state.worker = wrapped_->GetOrCreateWorker(target);
    if (state.worker != nullptr) {
      workers_.insert(std::make_pair(target, state));
    }
    return state.worker;
  }

  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "GetEagerClientCache");

    return wrapped_->GetEagerClientCache(eager_client_cache);
  }

  Status GetCoordinationClientCache(std::unique_ptr<CoordinationClientCache>*
                                        coordination_client_cache) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_6(mht_6_v, 259, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "GetCoordinationClientCache");

    return wrapped_->GetCoordinationClientCache(coordination_client_cache);
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "ReleaseWorker");

    // TODO(jeff,sanjay): Should decrement ref-count when we implement eviction.
  }

  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_8(mht_8_v, 276, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "GetDeviceLocalityNonBlocking");

    return wrapped_->GetDeviceLocalityNonBlocking(device, locality);
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_9(mht_9_v, 285, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "GetDeviceLocalityAsync");

    wrapped_->GetDeviceLocalityAsync(device, locality, done);
  }

  void SetLogging(bool active) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_10(mht_10_v, 292, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "SetLogging");
 wrapped_->SetLogging(active); }

  void ClearLogs() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_11(mht_11_v, 297, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "ClearLogs");
 wrapped_->ClearLogs(); }

  bool RetrieveLogs(int64_t step_id, StepStats* ss) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_12(mht_12_v, 302, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "RetrieveLogs");

    return wrapped_->RetrieveLogs(step_id, ss);
  }

 private:
  std::unique_ptr<WorkerCacheInterface> wrapped_;

  // Information kept per created WorkerInterface.
  struct WorkerState {
    WorkerInterface* worker;
    // TODO(jeff,sanjay): Add reference count if we support eviction.
  };

  // TODO(jeff,sanjay): Eviction when the map becomes too big.
  mutex mu_;
  std::unordered_map<string, WorkerState> workers_ TF_GUARDED_BY(mu_);
};

}  // namespace

WorkerSession::WorkerSession(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    std::unique_ptr<DeviceMgr> device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr)
    : session_name_(session_name),
      worker_name_(worker_name),
      worker_cache_(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr_(std::move(graph_mgr)),
      cluster_flr_(new ClusterFunctionLibraryRuntime(
          this, !session_name.empty(),
          remote_device_mgr ? remote_device_mgr.get() : nullptr)),
      device_mgr_(std::move(device_mgr)),
      borrowed_device_mgr_(nullptr),
      remote_device_mgr_(std::move(remote_device_mgr)) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("session_name: \"" + session_name + "\"");
   mht_13_v.push_back("worker_name: \"" + worker_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_13(mht_13_v, 341, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerSession::WorkerSession");

  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  worker_session_created->GetCell()->Set(true);
}

Status WorkerSession::UpdateWorkerCacheAndDevices(
    std::unique_ptr<WorkerCacheInterface> new_worker_cache,
    std::vector<std::unique_ptr<Device>> added_remote_devices,
    const std::vector<Device*>& removed_remote_devices) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_14(mht_14_v, 354, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerSession::UpdateWorkerCacheAndDevices");

  {
    mutex_lock l(worker_session_state_mu_);
    worker_cache_ = std::shared_ptr<WorkerCacheInterface>(
        new WorkerFreeListCache(std::move(new_worker_cache)));
  }
  TF_RETURN_IF_ERROR(remote_device_mgr_->RemoveDevices(removed_remote_devices));
  TF_RETURN_IF_ERROR(
      remote_device_mgr_->AddDevices(std::move(added_remote_devices)));
  return Status::OK();
}

/* static */
std::shared_ptr<WorkerSession> WorkerSession::CreateWithBorrowedDeviceMgr(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("session_name: \"" + session_name + "\"");
   mht_15_v.push_back("worker_name: \"" + worker_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_15(mht_15_v, 376, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerSession::CreateWithBorrowedDeviceMgr");

  return std::shared_ptr<WorkerSession>(new WorkerSession(
      session_name, worker_name, std::move(worker_cache), borrowed_device_mgr,
      std::move(graph_mgr), std::move(remote_device_mgr)));
}

WorkerSession::WorkerSession(
    const string& session_name, const string& worker_name,
    std::unique_ptr<WorkerCacheInterface> worker_cache,
    DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
    std::unique_ptr<DynamicDeviceMgr> remote_device_mgr)
    : session_name_(session_name),
      worker_name_(worker_name),
      worker_cache_(new WorkerFreeListCache(std::move(worker_cache))),
      graph_mgr_(std::move(graph_mgr)),
      cluster_flr_(new ClusterFunctionLibraryRuntime(
          this, !session_name.empty(), remote_device_mgr.get())),
      device_mgr_(nullptr),
      borrowed_device_mgr_(borrowed_device_mgr),
      remote_device_mgr_(std::move(remote_device_mgr)) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("session_name: \"" + session_name + "\"");
   mht_16_v.push_back("worker_name: \"" + worker_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_16(mht_16_v, 400, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerSession::WorkerSession");

  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  worker_session_created->GetCell()->Set(true);
}

WorkerSession::~WorkerSession() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTcc mht_17(mht_17_v, 410, "", "./tensorflow/core/distributed_runtime/worker_session.cc", "WorkerSession::~WorkerSession");

  if (graph_mgr_) {
    Status s = graph_mgr_->DeregisterAll();
    if (!s.ok()) {
      LOG(WARNING) << "Error during worker session deletion: " << s;
    }
  }
}

}  // namespace tensorflow
