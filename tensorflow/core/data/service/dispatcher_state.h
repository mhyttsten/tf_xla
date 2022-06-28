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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh() {
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


#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/auto_shard_rewriter.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// A class encapsulating the journaled state of the dispatcher. All state
// modifications must be done via `Apply`. This helps to ensure that
// replaying the journal will allow us to restore the exact same state.
//
// The following usage pattern will keep the journal in sync with the state of
// the dispatcher:
// {
//   mutex_lock l(mu_);
//   Update update = ...  // create an update
//   dispatcher_state.Apply(update);
//   journal_writer.write(Update);
//   // Unlock mu_
// }
//
// The division of functionality between DispatcherImpl and DispatcherState is
// as follows:
//   - DispatcherImpl is responsible for handling RPC requests, reading from
//     DispatcherState, and deciding what updates to apply to DispatcherState.
//     DispatcherImpl handles all synchronization.
//   - DispatcherState is responsible for making the state changes requested by
//     DispatcherImpl and for providing DispatcherImpl with read-only access to
//     the state.
//
// DispatcherState is thread-compatible but not thread-safe.
class DispatcherState {
 public:
  DispatcherState();
  explicit DispatcherState(
      const experimental::DispatcherConfig& dispatcher_config);
  DispatcherState(const DispatcherState&) = delete;
  DispatcherState& operator=(const DispatcherState&) = delete;

  // Applies the given update to the dispatcher's state.
  Status Apply(const Update& update);

  // A dataset registered with the dispatcher.
  struct Dataset {
    explicit Dataset(int64_t dataset_id, int64_t fingerprint,
                     const DataServiceMetadata& metadata)
        : dataset_id(dataset_id),
          fingerprint(fingerprint),
          metadata(metadata) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_0(mht_0_v, 250, "", "./tensorflow/core/data/service/dispatcher_state.h", "Dataset");
}

    const int64_t dataset_id;
    const int64_t fingerprint;
    const DataServiceMetadata metadata;
  };

  // A worker registered with the dispatcher.
  struct Worker {
    explicit Worker(const RegisterWorkerUpdate& register_worker)
        : address(register_worker.worker_address()),
          transfer_address(register_worker.transfer_address()),
          tags(register_worker.worker_tags().begin(),
               register_worker.worker_tags().end()),
          uid(register_worker.worker_uid()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_1(mht_1_v, 267, "", "./tensorflow/core/data/service/dispatcher_state.h", "Worker");
}

    const std::string address;
    const std::string transfer_address;
    const std::vector<std::string> tags;
    const int64_t uid;
  };

  // A key for identifying a job. The key contains a job name,
  // as well as a iteration number describing which iteration of the job we are
  // on.
  struct JobKey {
    explicit JobKey(absl::string_view name, int64_t iteration)
        : name(name), iteration(iteration) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_2(mht_2_v, 284, "", "./tensorflow/core/data/service/dispatcher_state.h", "JobKey");
}

    friend bool operator==(const JobKey& lhs, const JobKey& rhs) {
      return lhs.name == rhs.name && lhs.iteration == rhs.iteration;
    }

    template <typename H>
    friend H AbslHashValue(H h, const JobKey& k) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/data/service/dispatcher_state.h", "AbslHashValue");

      return H::combine(std::move(h), k.name, k.iteration);
    }

    std::string DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_4(mht_4_v, 301, "", "./tensorflow/core/data/service/dispatcher_state.h", "DebugString");

      return absl::StrCat(name, "/", iteration);
    }

    const std::string name;
    const int64_t iteration;
  };

  struct DistributedEpochState {
    explicit DistributedEpochState(int64_t num_split_providers)
        : iterations(num_split_providers), indices(num_split_providers) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_5(mht_5_v, 314, "", "./tensorflow/core/data/service/dispatcher_state.h", "DistributedEpochState");
}

    // The current iteration for each split provider.
    std::vector<int64_t> iterations;
    // Number of splits produced so far by each split provider.
    std::vector<int64_t> indices;
  };

  struct Task;

  struct PendingTask {
    explicit PendingTask(std::shared_ptr<Task> task, int64_t target_round)
        : task(std::move(task)), target_round(target_round) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_6(mht_6_v, 329, "", "./tensorflow/core/data/service/dispatcher_state.h", "PendingTask");
}

    std::shared_ptr<Task> task;
    // The target round where we want to insert the task.
    int64_t target_round;
    // Which consumers have responded that they have successfully blocked
    // before the target round.
    absl::flat_hash_set<int64_t> ready_consumers;
    // How many times we have failed to add the task.
    int64_t failures = 0;
  };

  // A job for processing a dataset.
  struct Job {
    explicit Job(int64_t job_id, int64_t dataset_id,
                 const ProcessingModeDef& processing_mode,
                 int64_t num_split_providers, JobKey job_key,
                 absl::optional<int64_t> num_consumers,
                 TargetWorkers target_workers)
        : job_id(job_id),
          dataset_id(dataset_id),
          processing_mode(processing_mode),
          job_key(job_key),
          num_consumers(num_consumers),
          target_workers(target_workers) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_7(mht_7_v, 356, "", "./tensorflow/core/data/service/dispatcher_state.h", "Job");

      if (IsDynamicShard(processing_mode)) {
        distributed_epoch_state = DistributedEpochState(num_split_providers);
      }
    }

    bool IsRoundRobin() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_8(mht_8_v, 365, "", "./tensorflow/core/data/service/dispatcher_state.h", "IsRoundRobin");
 return num_consumers.has_value(); }

    std::string DebugString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_9(mht_9_v, 370, "", "./tensorflow/core/data/service/dispatcher_state.h", "DebugString");

      return absl::StrCat(job_key.name, "_", job_key.iteration);
    }

    const int64_t job_id;
    const int64_t dataset_id;
    const ProcessingModeDef processing_mode;
    const JobKey job_key;
    absl::optional<DistributedEpochState> distributed_epoch_state;
    const absl::optional<int64_t> num_consumers;
    const TargetWorkers target_workers;
    std::queue<PendingTask> pending_tasks;
    int64_t num_clients = 0;
    int64_t last_client_released_micros = -1;
    bool finished = false;
    // Indicates whether the job was garbage collected.
    bool garbage_collected = false;
  };

  struct Task {
    template <class T>
    explicit Task(const T& create_task_update, const std::shared_ptr<Job>& job)
        : task_id(create_task_update.task_id()),
          job(job),
          worker_address(create_task_update.worker_address()),
          transfer_address(create_task_update.transfer_address()),
          worker_tags(create_task_update.worker_tags().begin(),
                      create_task_update.worker_tags().end()),
          worker_uid(create_task_update.worker_uid()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_stateDTh mht_10(mht_10_v, 401, "", "./tensorflow/core/data/service/dispatcher_state.h", "Task");
}

    const int64_t task_id;
    const std::shared_ptr<Job> job;
    const std::string worker_address;
    const std::string transfer_address;
    const std::vector<std::string> worker_tags;
    const int64_t worker_uid;
    int64_t starting_round = 0;
    bool finished = false;
    bool removed = false;
  };

  using TasksById = absl::flat_hash_map<int64_t, std::shared_ptr<Task>>;

  // Returns the next available dataset id.
  int64_t NextAvailableDatasetId() const;
  // Gets a dataset by id. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromId(int64_t id,
                       std::shared_ptr<const Dataset>& dataset) const;
  // Gets a dataset by fingerprint. Returns NOT_FOUND if there is no such
  // dataset.
  Status DatasetFromFingerprint(uint64 fingerprint,
                                std::shared_ptr<const Dataset>& dataset) const;

  // Gets a worker by address. Returns NOT_FOUND if there is no such worker.
  Status WorkerFromAddress(const std::string& address,
                           std::shared_ptr<const Worker>& worker) const;
  // Lists all workers registered with the dispatcher.
  std::vector<std::shared_ptr<const Worker>> ListWorkers() const;

  // Returns the next available job id.
  int64_t NextAvailableJobId() const;
  // Returns a list of all jobs.
  std::vector<std::shared_ptr<const Job>> ListJobs();
  // Gets a job by id. Returns NOT_FOUND if there is no such job.
  Status JobFromId(int64_t id, std::shared_ptr<const Job>& job) const;
  // Gets a job by key. Returns NOT_FOUND if there is no such job.
  Status JobByKey(JobKey key, std::shared_ptr<const Job>& job) const;

  // Returns the job associated with the given job client id. Returns NOT_FOUND
  // if the job_client_id is unknown or has been released.
  Status JobForJobClientId(int64_t job_client_id,
                           std::shared_ptr<const Job>& job);
  // Returns a list of all active client ids.
  std::vector<int64_t> ListActiveClientIds();
  // Returns the next available job client id.
  int64_t NextAvailableJobClientId() const;

  // Returns the next available task id.
  int64_t NextAvailableTaskId() const;
  // Gets a task by id. Returns NOT_FOUND if there is no such task.
  Status TaskFromId(int64_t id, std::shared_ptr<const Task>& task) const;
  // Stores a list of all tasks for the given job to `tasks`. Returns NOT_FOUND
  // if there is no such job.
  Status TasksForJob(int64_t job_id,
                     std::vector<std::shared_ptr<const Task>>& tasks) const;
  // Stores a list of all tasks for the given worker to `tasks`. Returns
  // NOT_FOUND if there is no such worker.
  Status TasksForWorker(const absl::string_view worker_address,
                        std::vector<std::shared_ptr<const Task>>& tasks) const;

  // If the dispatcher config explicitly specifies a list of workers, validates
  // `worker_address` is in the list.
  Status ValidateWorker(absl::string_view worker_address) const;

  // If the dispatcher config specifies worker addresses, `GetWorkerIndex`
  // returns the worker index according to the list. This is useful for
  // deterministically sharding a dataset among a fixed set of workers.
  StatusOr<int64_t> GetWorkerIndex(absl::string_view worker_address) const;

 private:
  void RegisterDataset(const RegisterDatasetUpdate& register_dataset);
  void RegisterWorker(const RegisterWorkerUpdate& register_worker);
  void CreateJob(const CreateJobUpdate& create_job);
  void ProduceSplit(const ProduceSplitUpdate& produce_split);
  void AcquireJobClient(const AcquireJobClientUpdate& acquire_job_client);
  void ReleaseJobClient(const ReleaseJobClientUpdate& release_job_client);
  void GarbageCollectJob(const GarbageCollectJobUpdate& garbage_collect_job);
  void RemoveTask(const RemoveTaskUpdate& remove_task);
  void CreatePendingTask(const CreatePendingTaskUpdate& create_pending_task);
  void ClientHeartbeat(const ClientHeartbeatUpdate& client_heartbeat);
  void CreateTask(const CreateTaskUpdate& create_task);
  void FinishTask(const FinishTaskUpdate& finish_task);

  int64_t next_available_dataset_id_ = 1000;
  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Dataset>> datasets_by_id_;
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>>
      datasets_by_fingerprint_;

  // Registered workers, keyed by address.
  absl::flat_hash_map<std::string, std::shared_ptr<Worker>> workers_;

  // Assigns an index to each worker according to worker addresses list
  // specified in the dispatcher config.
  WorkerIndexResolver worker_index_resolver_;

  int64_t next_available_job_id_ = 2000;
  // Jobs, keyed by job ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Job>> jobs_;
  // Jobs, keyed by their job keys.
  absl::flat_hash_map<JobKey, std::shared_ptr<Job>> jobs_by_key_;

  int64_t next_available_job_client_id_ = 3000;
  // Mapping from client ids to the jobs they are associated with.
  absl::flat_hash_map<int64_t, std::shared_ptr<Job>> jobs_for_client_ids_;

  int64_t next_available_task_id_ = 4000;
  // Tasks, keyed by task ids.
  TasksById tasks_;
  // List of tasks associated with each job.
  absl::flat_hash_map<int64_t, std::vector<std::shared_ptr<Task>>>
      tasks_by_job_;
  // Tasks, keyed by worker addresses. The values are a map from task id to
  // task.
  absl::flat_hash_map<std::string, TasksById> tasks_by_worker_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
