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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTh() {
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
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace data {

// A TensorFlow DataService serves dataset elements over RPC.
class DataServiceWorkerImpl {
 public:
  explicit DataServiceWorkerImpl(const experimental::WorkerConfig& config);
  ~DataServiceWorkerImpl();

  // Starts the worker. The worker needs to know its own address so that it can
  // register with the dispatcher. This is set in `Start` instead of in the
  // constructor because the worker may be binding to port `0`, in which case
  // the address isn't known until the worker has started and decided which port
  // to bind to.
  Status Start(const std::string& worker_address,
               const std::string& transfer_address);
  // Stops the worker, attempting a clean shutdown by rejecting new requests
  // and waiting for outstanding requests to complete.
  void Stop();

  // Serves a GetElement request, storing the result in `*result`. See
  // worker.proto for GetElement API documentation.
  Status GetElementResult(const GetElementRequest* request,
                          GetElementResult* result);

  // Deletes the local task and iterator. Only called by local clients to delete
  // unused task iterators assuming the task is not read by remote clients. This
  // method is not visible to gRPC clients.
  void DeleteLocalTask(const TaskInfo& task_info);

  // See worker.proto for API documentation.

  /// Dispatcher-facing API.
  Status ProcessTask(const ProcessTaskRequest* request,
                     ProcessTaskResponse* response);

  /// Client-facing API.
  Status GetElement(const GetElementRequest* request,
                    GetElementResponse* response);
  Status GetWorkerTasks(const GetWorkerTasksRequest* request,
                        GetWorkerTasksResponse* response);

 private:
  struct Task {
    explicit Task(TaskDef task_def) : task_def(std::move(task_def)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTh mht_0(mht_0_v, 255, "", "./tensorflow/core/data/service/worker_impl.h", "Task");
}

    TaskDef task_def;
    mutex mu;
    bool initialized TF_GUARDED_BY(mu) = false;
    int64_t outstanding_requests TF_GUARDED_BY(&DataServiceWorkerImpl::mu_) = 0;
    std::unique_ptr<TaskRunner> task_runner;
  };

  // Validates the worker config.
  Status ValidateWorkerConfig() const;
  // Sends task status to the dispatcher and checks for dispatcher commands.
  Status SendTaskUpdates() TF_LOCKS_EXCLUDED(mu_);
  // Creates an iterator to process a task.
  Status ProcessTaskInternal(const TaskDef& task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status EnsureTaskInitialized(Task& task);
  // Stops a task, cancelling the task's outstanding requests and waiting for
  // them to finish.
  void StopTask(Task& task) TF_LOCKS_EXCLUDED(mu_);
  // A thread for notifying the dispatcher when tasks complete.
  void TaskCompletionThread() TF_LOCKS_EXCLUDED(mu_);
  // A thread for doing periodic heartbeats to the dispatcher.
  void HeartbeatThread() TF_LOCKS_EXCLUDED(mu_);
  // Performs a heartbeat to the dispatcher.
  Status Heartbeat() TF_LOCKS_EXCLUDED(mu_);
  // Gets the DatasetDef for `task_def`.
  StatusOr<DatasetDef> GetDatasetDef(const TaskDef& task_def) const;
  // Creates a dataset from `dataset_def`.
  StatusOr<std::unique_ptr<standalone::Dataset>> MakeDataset(
      const DatasetDef& dataset_def, const TaskDef& task_def) const;
  // Creates an iterator for `dataset`.
  StatusOr<std::unique_ptr<standalone::Iterator>> MakeDatasetIterator(
      standalone::Dataset& dataset, const TaskDef& task_def) const;

  const experimental::WorkerConfig config_;
  // Worker Borg job UID for telemetry. -1 if not supported.
  const int64_t worker_uid_;

  // The worker's own address.
  std::string worker_address_;
  std::string transfer_address_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_;

  mutex mu_;
  condition_variable cv_;
  // Information about tasks, keyed by task ids. The tasks are updated based on
  // the heartbeat responses from the dispatcher.
  absl::flat_hash_map<int64_t, std::shared_ptr<Task>> tasks_ TF_GUARDED_BY(mu_);
  // Ids of tasks that have finished.
  absl::flat_hash_set<int64_t> finished_tasks_ TF_GUARDED_BY(mu_);
  // Completed tasks which haven't yet been communicated to the dispatcher.
  absl::flat_hash_set<int64_t> pending_completed_tasks_ TF_GUARDED_BY(mu_);
  // Tasks deleted by the local client. If the client tries to read from them
  // again, the worker will return a non-retriable FailedPrecondition error.
  absl::flat_hash_set<int64_t> deleted_tasks_ TF_GUARDED_BY(mu_);
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Whether the worker has registered with the dispatcher yet.
  bool registered_ TF_GUARDED_BY(mu_) = false;
  // A thread for notifying the dispatcher when tasks complete.
  std::unique_ptr<Thread> task_completion_thread_;
  condition_variable task_completion_cv_ TF_GUARDED_BY(mu_);
  // A thread for performing regular heartbeats to the dispatcher.
  std::unique_ptr<Thread> heartbeat_thread_;
  condition_variable heartbeat_cv_ TF_GUARDED_BY(mu_);
  int64_t outstanding_requests_ TF_GUARDED_BY(mu_) = 0;
  CancellationManager cancellation_manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceWorkerImpl);
};

// Local in-process workers shared among clients and servers. If clients and
// workers colocate in the same process, clients can read from local workers to
// reduce RPC calls and data copy.
class LocalWorkers {
 public:
  // Adds a `worker` at `worker_address`. If a worker already exists at the
  // address, it will be updated to the new `worker`.
  // REQUIRES: worker != nullptr.
  static void Add(absl::string_view worker_address,
                  std::shared_ptr<DataServiceWorkerImpl> worker);

  // Gets a local worker at `worker_address`. Returns nullptr if a worker is not
  // found.
  static std::shared_ptr<DataServiceWorkerImpl> Get(
      absl::string_view worker_address);

  // Returns if there are any local workers in the process.
  static bool Empty();

  // Removes a worker at `worker_address`. It is no-op if a worker is not found
  // at the address.
  static void Remove(absl::string_view worker_address);

 private:
  using AddressToWorkerMap =
      absl::flat_hash_map<std::string, std::shared_ptr<DataServiceWorkerImpl>>;
  static mutex mu_;
  static AddressToWorkerMap* local_workers_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
