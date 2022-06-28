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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc() {
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
#include "tensorflow/core/data/service/dispatcher_state.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

DispatcherState::DispatcherState()
    : worker_index_resolver_(std::vector<std::string>{}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::DispatcherState");
}

DispatcherState::DispatcherState(
    const experimental::DispatcherConfig& dispatcher_config)
    : worker_index_resolver_(dispatcher_config.worker_addresses()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::DispatcherState");
}

Status DispatcherState::Apply(const Update& update) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::Apply");

  switch (update.update_type_case()) {
    case Update::kRegisterDataset:
      RegisterDataset(update.register_dataset());
      break;
    case Update::kRegisterWorker:
      RegisterWorker(update.register_worker());
      break;
    case Update::kCreateJob:
      CreateJob(update.create_job());
      break;
    case Update::kProduceSplit:
      ProduceSplit(update.produce_split());
      break;
    case Update::kAcquireJobClient:
      AcquireJobClient(update.acquire_job_client());
      break;
    case Update::kReleaseJobClient:
      ReleaseJobClient(update.release_job_client());
      break;
    case Update::kGarbageCollectJob:
      GarbageCollectJob(update.garbage_collect_job());
      break;
    case Update::kRemoveTask:
      RemoveTask(update.remove_task());
      break;
    case Update::kCreatePendingTask:
      CreatePendingTask(update.create_pending_task());
      break;
    case Update::kClientHeartbeat:
      ClientHeartbeat(update.client_heartbeat());
      break;
    case Update::kCreateTask:
      CreateTask(update.create_task());
      break;
    case Update::kFinishTask:
      FinishTask(update.finish_task());
      break;
    case Update::UPDATE_TYPE_NOT_SET:
      return errors::Internal("Update type not set.");
  }

  return Status::OK();
}

void DispatcherState::RegisterDataset(
    const RegisterDatasetUpdate& register_dataset) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::RegisterDataset");

  int64_t id = register_dataset.dataset_id();
  int64_t fingerprint = register_dataset.fingerprint();
  auto dataset =
      std::make_shared<Dataset>(id, fingerprint, register_dataset.metadata());
  DCHECK(!datasets_by_id_.contains(id));
  datasets_by_id_[id] = dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[fingerprint] = dataset;
  next_available_dataset_id_ = std::max(next_available_dataset_id_, id + 1);
}

void DispatcherState::RegisterWorker(
    const RegisterWorkerUpdate& register_worker) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_4(mht_4_v, 283, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::RegisterWorker");

  std::string address = register_worker.worker_address();
  DCHECK(!workers_.contains(address));
  workers_[address] = std::make_shared<Worker>(register_worker);
  tasks_by_worker_[address] =
      absl::flat_hash_map<int64_t, std::shared_ptr<Task>>();
  worker_index_resolver_.AddWorker(address);
}

void DispatcherState::CreateJob(const CreateJobUpdate& create_job) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_5(mht_5_v, 295, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::CreateJob");

  int64_t job_id = create_job.job_id();
  JobKey job_key(create_job.job_key().name(), create_job.job_key().iteration());
  absl::optional<int64_t> num_consumers;
  if (create_job.optional_num_consumers_case() ==
      CreateJobUpdate::kNumConsumers) {
    num_consumers = create_job.num_consumers();
  }
  auto job = std::make_shared<Job>(job_id, create_job.dataset_id(),
                                   create_job.processing_mode_def(),
                                   create_job.num_split_providers(), job_key,
                                   num_consumers, create_job.target_workers());
  DCHECK(!jobs_.contains(job_id));
  jobs_[job_id] = job;
  tasks_by_job_[job_id] = std::vector<std::shared_ptr<Task>>();
  DCHECK(!jobs_by_key_.contains(job_key) ||
         jobs_by_key_[job_key]->garbage_collected);
  jobs_by_key_[job_key] = job;
  next_available_job_id_ = std::max(next_available_job_id_, job_id + 1);
}

void DispatcherState::ProduceSplit(const ProduceSplitUpdate& produce_split) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_6(mht_6_v, 319, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ProduceSplit");

  std::shared_ptr<Job> job = jobs_[produce_split.job_id()];
  DCHECK(job->distributed_epoch_state.has_value());
  DistributedEpochState& state = job->distributed_epoch_state.value();
  int64_t provider_index = produce_split.split_provider_index();
  DCHECK_EQ(produce_split.iteration(), state.iterations[provider_index]);
  if (produce_split.finished()) {
    state.iterations[provider_index]++;
    state.indices[provider_index] = 0;
    return;
  }
  state.indices[provider_index]++;
}

void DispatcherState::AcquireJobClient(
    const AcquireJobClientUpdate& acquire_job_client) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_7(mht_7_v, 337, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::AcquireJobClient");

  int64_t job_client_id = acquire_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job);
  job = jobs_[acquire_job_client.job_id()];
  DCHECK(job);
  job->num_clients++;
  next_available_job_client_id_ =
      std::max(next_available_job_client_id_, job_client_id + 1);
}

void DispatcherState::ReleaseJobClient(
    const ReleaseJobClientUpdate& release_job_client) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_8(mht_8_v, 352, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ReleaseJobClient");

  int64_t job_client_id = release_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(job);
  job->num_clients--;
  DCHECK_GE(job->num_clients, 0);
  job->last_client_released_micros = release_job_client.time_micros();
  jobs_for_client_ids_.erase(job_client_id);
}

void DispatcherState::GarbageCollectJob(
    const GarbageCollectJobUpdate& garbage_collect_job) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_9(mht_9_v, 366, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::GarbageCollectJob");

  int64_t job_id = garbage_collect_job.job_id();
  for (auto& task : tasks_by_job_[job_id]) {
    task->finished = true;
    tasks_by_worker_[task->worker_address].erase(task->task_id);
  }
  jobs_[job_id]->finished = true;
  jobs_[job_id]->garbage_collected = true;
}

void DispatcherState::RemoveTask(const RemoveTaskUpdate& remove_task) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_10(mht_10_v, 379, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::RemoveTask");

  std::shared_ptr<Task>& task = tasks_[remove_task.task_id()];
  DCHECK(task);
  task->removed = true;
  auto& tasks_for_job = tasks_by_job_[task->job->job_id];
  for (auto it = tasks_for_job.begin(); it != tasks_for_job.end(); ++it) {
    if ((*it)->task_id == task->task_id) {
      tasks_for_job.erase(it);
      break;
    }
  }
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  tasks_.erase(task->task_id);
  VLOG(1) << "Removed task " << remove_task.task_id() << " from worker "
          << task->worker_address;
}

void DispatcherState::CreatePendingTask(
    const CreatePendingTaskUpdate& create_pending_task) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_11(mht_11_v, 400, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::CreatePendingTask");

  int64_t task_id = create_pending_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_pending_task.job_id()];
  DCHECK_NE(job, nullptr);
  task = std::make_shared<Task>(create_pending_task, job);
  job->pending_tasks.emplace(task, create_pending_task.starting_round());
  tasks_by_worker_[create_pending_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::ClientHeartbeat(
    const ClientHeartbeatUpdate& client_heartbeat) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_12(mht_12_v, 416, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ClientHeartbeat");

  int64_t job_client_id = client_heartbeat.job_client_id();
  auto& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job->pending_tasks.empty());
  auto& task = job->pending_tasks.front();
  if (client_heartbeat.has_task_rejected()) {
    task.failures++;
    task.ready_consumers.clear();
    task.target_round = client_heartbeat.task_rejected().new_target_round();
  }
  if (client_heartbeat.task_accepted()) {
    task.ready_consumers.insert(job_client_id);
    if (task.ready_consumers.size() == job->num_consumers.value()) {
      VLOG(1) << "Promoting task " << task.task->task_id
              << " from pending to active";
      task.task->starting_round = task.target_round;
      tasks_by_job_[job->job_id].push_back(task.task);
      job->pending_tasks.pop();
    }
  }
}

void DispatcherState::CreateTask(const CreateTaskUpdate& create_task) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_13(mht_13_v, 441, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::CreateTask");

  int64_t task_id = create_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_task.job_id()];
  DCHECK_NE(job, nullptr);
  task = std::make_shared<Task>(create_task, job);
  tasks_by_job_[create_task.job_id()].push_back(task);
  tasks_by_worker_[create_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::FinishTask(const FinishTaskUpdate& finish_task) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_14(mht_14_v, 456, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::FinishTask");

  VLOG(2) << "Marking task " << finish_task.task_id() << " as finished";
  int64_t task_id = finish_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK(task != nullptr);
  task->finished = true;
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  bool all_finished = true;
  for (const auto& task_for_job : tasks_by_job_[task->job->job_id]) {
    if (!task_for_job->finished) {
      all_finished = false;
    }
  }
  VLOG(3) << "Job " << task->job->job_id << " finished: " << all_finished;
  jobs_[task->job->job_id]->finished = all_finished;
}

int64_t DispatcherState::NextAvailableDatasetId() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_15(mht_15_v, 476, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::NextAvailableDatasetId");

  return next_available_dataset_id_;
}

Status DispatcherState::DatasetFromId(
    int64_t id, std::shared_ptr<const Dataset>& dataset) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_16(mht_16_v, 484, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::DatasetFromId");

  auto it = datasets_by_id_.find(id);
  if (it == datasets_by_id_.end()) {
    return errors::NotFound("Dataset id ", id, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::DatasetFromFingerprint(
    uint64 fingerprint, std::shared_ptr<const Dataset>& dataset) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_17(mht_17_v, 497, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::DatasetFromFingerprint");

  auto it = datasets_by_fingerprint_.find(fingerprint);
  if (it == datasets_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::WorkerFromAddress(
    const std::string& address, std::shared_ptr<const Worker>& worker) const {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("address: \"" + address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_18(mht_18_v, 511, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::WorkerFromAddress");

  auto it = workers_.find(address);
  if (it == workers_.end()) {
    return errors::NotFound("Worker with address ", address, " not found.");
  }
  worker = it->second;
  return Status::OK();
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ListWorkers() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_19(mht_19_v, 524, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ListWorkers");

  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(workers_.size());
  for (const auto& it : workers_) {
    workers.push_back(it.second);
  }
  return workers;
}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobs() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_20(mht_20_v, 537, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ListJobs");

  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  jobs.reserve(jobs_.size());
  for (const auto& it : jobs_) {
    jobs.push_back(it.second);
  }
  return jobs;
}

Status DispatcherState::JobFromId(int64_t id,
                                  std::shared_ptr<const Job>& job) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_21(mht_21_v, 550, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::JobFromId");

  auto it = jobs_.find(id);
  if (it == jobs_.end()) {
    return errors::NotFound("Job id ", id, " not found");
  }
  job = it->second;
  return Status::OK();
}

Status DispatcherState::JobByKey(JobKey job_key,
                                 std::shared_ptr<const Job>& job) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_22(mht_22_v, 563, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::JobByKey");

  auto it = jobs_by_key_.find(job_key);
  if (it == jobs_by_key_.end()) {
    return errors::NotFound("Job key (", job_key.name, ", ", job_key.iteration,
                            ") not found");
  }
  job = it->second;
  return Status::OK();
}

int64_t DispatcherState::NextAvailableJobId() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_23(mht_23_v, 576, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::NextAvailableJobId");

  return next_available_job_id_;
}

Status DispatcherState::JobForJobClientId(int64_t job_client_id,
                                          std::shared_ptr<const Job>& job) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_24(mht_24_v, 584, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::JobForJobClientId");

  job = jobs_for_client_ids_[job_client_id];
  if (!job) {
    return errors::NotFound("Job client id not found: ", job_client_id);
  }
  return Status::OK();
}

std::vector<int64_t> DispatcherState::ListActiveClientIds() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_25(mht_25_v, 595, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ListActiveClientIds");

  std::vector<int64_t> ids;
  for (const auto& it : jobs_for_client_ids_) {
    if (it.second && !it.second->finished) {
      ids.push_back(it.first);
    }
  }
  return ids;
}

int64_t DispatcherState::NextAvailableJobClientId() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_26(mht_26_v, 608, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::NextAvailableJobClientId");

  return next_available_job_client_id_;
}

Status DispatcherState::TaskFromId(int64_t id,
                                   std::shared_ptr<const Task>& task) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_27(mht_27_v, 616, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::TaskFromId");

  auto it = tasks_.find(id);
  if (it == tasks_.end()) {
    return errors::NotFound("Task ", id, " not found");
  }
  task = it->second;
  return Status::OK();
}

Status DispatcherState::TasksForJob(
    int64_t job_id, std::vector<std::shared_ptr<const Task>>& tasks) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_28(mht_28_v, 629, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::TasksForJob");

  auto it = tasks_by_job_.find(job_id);
  if (it == tasks_by_job_.end()) {
    return errors::NotFound("Job ", job_id, " not found");
  }
  tasks.clear();
  tasks.reserve(it->second.size());
  for (const auto& task : it->second) {
    tasks.push_back(task);
  }
  return Status::OK();
}

Status DispatcherState::TasksForWorker(
    absl::string_view worker_address,
    std::vector<std::shared_ptr<const Task>>& tasks) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_29(mht_29_v, 648, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::TasksForWorker");

  tasks.clear();
  auto it = tasks_by_worker_.find(worker_address);
  if (it == tasks_by_worker_.end()) {
    return errors::NotFound("Worker ", worker_address, " not found");
  }
  const absl::flat_hash_map<int64_t, std::shared_ptr<Task>>& worker_tasks =
      it->second;
  tasks.reserve(worker_tasks.size());
  for (const auto& task : worker_tasks) {
    tasks.push_back(task.second);
  }
  return Status::OK();
}

int64_t DispatcherState::NextAvailableTaskId() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_30(mht_30_v, 666, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::NextAvailableTaskId");

  return next_available_task_id_;
}

Status DispatcherState::ValidateWorker(absl::string_view worker_address) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_31(mht_31_v, 674, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::ValidateWorker");

  return worker_index_resolver_.ValidateWorker(worker_address);
}

StatusOr<int64_t> DispatcherState::GetWorkerIndex(
    absl::string_view worker_address) const {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTcc mht_32(mht_32_v, 683, "", "./tensorflow/core/data/service/dispatcher_state.cc", "DispatcherState::GetWorkerIndex");

  return worker_index_resolver_.GetWorkerIndex(worker_address);
}

}  // namespace data
}  // namespace tensorflow
