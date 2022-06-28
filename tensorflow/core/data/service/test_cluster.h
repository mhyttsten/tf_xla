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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh() {
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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_client.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// Helper class for unit testing a tf.data service cluster.
class TestCluster {
 public:
  struct Config {
   public:
    int num_workers = 3;
    int64_t client_timeout_ms = 0;
    int64_t job_gc_check_interval_ms = 0;
    int64_t job_gc_timeout_ms = 0;
  };

  // Creates a new test cluster with a dispatcher and `num_workers` workers.
  explicit TestCluster(int num_workers);
  explicit TestCluster(const Config& config);

  // Initializes the test cluster. This must be called before interacting with
  // the cluster. Initialize should be called only once.
  Status Initialize();
  // Adds a new worker to the cluster.
  Status AddWorker();
  // Returns the number of workers in this cluster.
  size_t NumWorkers() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_0(mht_0_v, 233, "", "./tensorflow/core/data/service/test_cluster.h", "NumWorkers");
 return workers_.size(); }
  // Returns the number of active jobs.
  StatusOr<size_t> NumActiveJobs() const {
    return dispatcher_->NumActiveJobs();
  }
  // Returns the dispatcher address in the form "hostname:port".
  std::string DispatcherAddress() const;
  // Returns the address of the worker at the specified index, in the form
  // "hostname:port". The index must be non-negative and less than the number of
  // workers in the cluster.
  std::string WorkerAddress(int index) const;

  // Stops one worker.
  void StopWorker(size_t index);
  // Stops all workers.
  void StopWorkers();

 private:
  bool initialized_ = false;
  int num_workers_;
  Config config_;
  std::unique_ptr<DispatchGrpcDataServer> dispatcher_;
  std::string dispatcher_address_;
  std::vector<std::unique_ptr<WorkerGrpcDataServer>> workers_;
  std::vector<std::string> worker_addresses_;
};

// A test utility to provide a `DatasetDef` to a `TestCluster` and generate data
// from each worker for verification. For example:
//
// TestCluster cluster(/*num_workers=*/2);
// TF_ASSERT_OK(cluster.Initialize());
// DatasetClient<int64_t> dataset_reader(cluster);
//
// EXPECT_THAT(
//     dataset_reader.Read(RangeDataset(4), ProcessingModeDef::DATA,
//                         TARGET_WORKERS_LOCAL),
//     IsOkAndHolds(UnorderedElementsAre(
//         Pair(cluster.WorkerAddress(0), ElementsAre(0, 2)),
//         Pair(cluster.WorkerAddress(1), ElementsAre(1, 3)))));
template <class T>
class DatasetClient {
 public:
  // Creates a dataset client. It will process datasets in `cluster`.
  explicit DatasetClient(const TestCluster& cluster);

  // Maps a worker address to the data it produces when calling `Read`.
  using WorkerResultMap = absl::flat_hash_map<std::string, std::vector<T>>;

  // Processes `dataset` and retrieves the data from workers. Returns the data
  // produced by each worker, keyed by the worker address.
  StatusOr<WorkerResultMap> Read(
      const DatasetDef& dataset,
      ProcessingModeDef::ShardingPolicy sharding_policy,
      TargetWorkers target_workers);
  // Creates a job and returns the job client ID.
  StatusOr<int64_t> CreateJob();
  // Gets the tasks for job `job_client_id`. The job has one task processed by
  // every worker.
  StatusOr<std::vector<TaskInfo>> GetTasks(int64 job_client_id);

 private:
  // Registers the dataset and returns the dataset ID.
  StatusOr<int64_t> RegisterDataset(const DatasetDef& dataset);
  // Creates a job and returns the job client ID.
  StatusOr<int64_t> CreateJob(int64 dataset_id,
                              ProcessingModeDef::ShardingPolicy sharding_policy,
                              TargetWorkers target_workers);
  // Reads values from `tasks`, one task at a time, until all tasks have
  // finished.
  StatusOr<WorkerResultMap> ReadFromTasks(const std::vector<TaskInfo>& tasks);
  // Reads the next element from the specified task.
  StatusOr<GetElementResult> ReadFromTask(const TaskInfo& task_info);

  const TestCluster& cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
  absl::flat_hash_map<std::string, std::unique_ptr<DataServiceWorkerClient>>
      worker_clients_;
};

template <class T>
DatasetClient<T>::DatasetClient(const TestCluster& cluster)
    : cluster_(cluster) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_1(mht_1_v, 318, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::DatasetClient");

  dispatcher_client_ = absl::make_unique<DataServiceDispatcherClient>(
      cluster_.DispatcherAddress(), "grpc");

  for (size_t i = 0; i < cluster.NumWorkers(); ++i) {
    worker_clients_[cluster_.WorkerAddress(i)] =
        absl::make_unique<DataServiceWorkerClient>(cluster_.WorkerAddress(i),
                                                   "grpc", "grpc");
  }
}

template <class T>
StatusOr<typename DatasetClient<T>::WorkerResultMap> DatasetClient<T>::Read(
    const DatasetDef& dataset,
    ProcessingModeDef::ShardingPolicy sharding_policy,
    TargetWorkers target_workers) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_2(mht_2_v, 336, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::Read");

  TF_ASSIGN_OR_RETURN(const int64 dataset_id, RegisterDataset(dataset));
  TF_ASSIGN_OR_RETURN(const int64 job_client_id,
                      CreateJob(dataset_id, sharding_policy, target_workers));
  TF_ASSIGN_OR_RETURN(const std::vector<TaskInfo> tasks,
                      GetTasks(job_client_id));
  return ReadFromTasks(tasks);
}

template <class T>
StatusOr<int64_t> DatasetClient<T>::RegisterDataset(const DatasetDef& dataset) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_3(mht_3_v, 349, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::RegisterDataset");

  int64 dataset_id = 0;
  TF_RETURN_IF_ERROR(dispatcher_client_->RegisterDataset(
      dataset, DataServiceMetadata(), dataset_id));
  return dataset_id;
}

template <class T>
StatusOr<int64_t> DatasetClient<T>::CreateJob(
    const int64 dataset_id, ProcessingModeDef::ShardingPolicy sharding_policy,
    TargetWorkers target_workers) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_4(mht_4_v, 362, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::CreateJob");

  int64 job_client_id = 0;
  ProcessingModeDef processing_mode_def;
  processing_mode_def.set_sharding_policy(sharding_policy);
  TF_RETURN_IF_ERROR(dispatcher_client_->GetOrCreateJob(
      dataset_id, processing_mode_def, /*job_key=*/absl::nullopt,
      /*num_consumers=*/absl::nullopt, target_workers, job_client_id));
  return job_client_id;
}

template <class T>
StatusOr<int64_t> DatasetClient<T>::CreateJob() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_5(mht_5_v, 376, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::CreateJob");

  TF_ASSIGN_OR_RETURN(
      const int64 dataset_id,
      RegisterDataset(tensorflow::data::testing::RangeDataset(10)));
  return CreateJob(dataset_id, ProcessingModeDef::OFF, TARGET_WORKERS_ANY);
}

template <class T>
StatusOr<std::vector<TaskInfo>> DatasetClient<T>::GetTasks(
    const int64 job_client_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_6(mht_6_v, 388, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::GetTasks");

  ClientHeartbeatRequest request;
  ClientHeartbeatResponse response;
  request.set_job_client_id(job_client_id);
  TF_RETURN_IF_ERROR(dispatcher_client_->ClientHeartbeat(request, response));
  if (response.task_info().empty()) {
    return errors::NotFound("No task found for job ", job_client_id, ".");
  }
  return std::vector<TaskInfo>(response.task_info().begin(),
                               response.task_info().end());
}

template <class T>
StatusOr<typename DatasetClient<T>::WorkerResultMap>
DatasetClient<T>::ReadFromTasks(const std::vector<TaskInfo>& tasks) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_7(mht_7_v, 405, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::ReadFromTasks");

  WorkerResultMap result;
  bool all_workers_finished = false;
  while (!all_workers_finished) {
    all_workers_finished = true;
    for (const TaskInfo& task : tasks) {
      StatusOr<GetElementResult> element_result = ReadFromTask(task);
      // A task may be cancelled when it has finished but other workers are
      // still producing data.
      if (errors::IsCancelled(element_result.status())) {
        continue;
      }
      TF_RETURN_IF_ERROR(element_result.status());
      if (element_result->end_of_sequence) {
        continue;
      }
      all_workers_finished = false;
      result[task.worker_address()].push_back(
          element_result->components[0].unaligned_flat<T>().data()[0]);
    }
  }
  return result;
}

template <class T>
StatusOr<GetElementResult> DatasetClient<T>::ReadFromTask(
    const TaskInfo& task_info) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTh mht_8(mht_8_v, 434, "", "./tensorflow/core/data/service/test_cluster.h", "DatasetClient<T>::ReadFromTask");

  GetElementRequest request;
  GetElementResult element_result;
  request.set_task_id(task_info.task_id());
  TF_RETURN_IF_ERROR(worker_clients_[task_info.worker_address()]->GetElement(
      request, element_result));
  return element_result;
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_TEST_CLUSTER_H_
