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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc() {
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

#include <memory>
#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

namespace {
using Dataset = DispatcherState::Dataset;
using Worker = DispatcherState::Worker;
using JobKey = DispatcherState::JobKey;
using Job = DispatcherState::Job;
using Task = DispatcherState::Task;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

Status RegisterDataset(int64_t id, uint64 fingerprint, DispatcherState& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "RegisterDataset");

  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(id);
  register_dataset->set_fingerprint(fingerprint);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status RegisterDataset(int64_t id, DispatcherState& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "RegisterDataset");

  return RegisterDataset(id, /*fingerprint=*/1, state);
}

Status RegisterWorker(std::string worker_address, DispatcherState& state) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("worker_address: \"" + worker_address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "RegisterWorker");

  Update update;
  update.mutable_register_worker()->set_worker_address(worker_address);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status CreateJob(int64_t job_id, int64_t dataset_id,
                 const JobKey& named_job_key, DispatcherState& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "CreateJob");

  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->mutable_processing_mode_def()->set_sharding_policy(
      ProcessingModeDef::OFF);
  JobKeyDef* key = create_job->mutable_job_key();
  key->set_name(named_job_key.name);
  key->set_iteration(named_job_key.iteration);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status CreateJob(int64_t job_id, int64_t dataset_id, DispatcherState& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "CreateJob");

  JobKey key(/*name=*/absl::StrCat(random::New64()), /*iteration=*/0);
  return CreateJob(job_id, dataset_id, key, state);
}

Status AcquireJobClientId(int64_t job_id, int64_t job_client_id,
                          DispatcherState& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "AcquireJobClientId");

  Update update;
  AcquireJobClientUpdate* acquire_job_client =
      update.mutable_acquire_job_client();
  acquire_job_client->set_job_id(job_id);
  acquire_job_client->set_job_client_id(job_client_id);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status ReleaseJobClientId(int64_t job_client_id, int64_t release_time,
                          DispatcherState& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_6(mht_6_v, 288, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "ReleaseJobClientId");

  Update update;
  ReleaseJobClientUpdate* release_job_client =
      update.mutable_release_job_client();
  release_job_client->set_job_client_id(job_client_id);
  release_job_client->set_time_micros(release_time);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status CreateTask(int64_t task_id, int64_t job_id,
                  const std::string& worker_address, DispatcherState& state) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("worker_address: \"" + worker_address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_7(mht_7_v, 303, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "CreateTask");

  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job_id);
  create_task->set_worker_address(worker_address);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}

Status FinishTask(int64_t task_id, DispatcherState& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_state_testDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/data/service/dispatcher_state_test.cc", "FinishTask");

  Update update;
  FinishTaskUpdate* finish_task = update.mutable_finish_task();
  finish_task->set_task_id(task_id);
  TF_RETURN_IF_ERROR(state.Apply(update));
  return Status::OK();
}
}  // namespace

TEST(DispatcherState, RegisterDataset) {
  uint64 fingerprint = 20;
  DispatcherState state;
  int64_t id = state.NextAvailableDatasetId();
  TF_EXPECT_OK(RegisterDataset(id, fingerprint, state));
  EXPECT_EQ(state.NextAvailableDatasetId(), id + 1);

  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, dataset));
    EXPECT_EQ(dataset->dataset_id, id);
    EXPECT_TRUE(dataset->metadata.element_spec().empty());
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_UNSPECIFIED);
  }
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(id, dataset));
    EXPECT_EQ(dataset->fingerprint, fingerprint);
    EXPECT_TRUE(dataset->metadata.element_spec().empty());
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_UNSPECIFIED);
  }
}

TEST(DispatcherState, RegisterDatasetCompression) {
  DispatcherState state;
  const int64_t dataset_id = state.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->mutable_metadata()->set_compression(
      DataServiceMetadata::COMPRESSION_SNAPPY);
  TF_ASSERT_OK(state.Apply(update));
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(dataset_id, dataset));
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_SNAPPY);
  }
}

TEST(DispatcherState, RegisterDatasetElementSpec) {
  DispatcherState state;
  const int64_t dataset_id = state.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(20);
  register_dataset->mutable_metadata()->set_element_spec(
      "encoded_element_spec");
  TF_ASSERT_OK(state.Apply(update));
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(dataset_id, dataset));
    EXPECT_EQ(dataset->metadata.element_spec(), "encoded_element_spec");
  }
}

TEST(DispatcherState, MissingDatasetId) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromId(0, dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, MissingDatasetFingerprint) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromFingerprint(0, dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, NextAvailableDatasetId) {
  DispatcherState state;
  int64_t id = state.NextAvailableDatasetId();
  uint64 fingerprint = 20;
  TF_EXPECT_OK(RegisterDataset(id, fingerprint, state));
  EXPECT_NE(state.NextAvailableDatasetId(), id);
  EXPECT_EQ(state.NextAvailableDatasetId(), state.NextAvailableDatasetId());
}

TEST(DispatcherState, RegisterWorker) {
  DispatcherState state;
  std::string address = "test_worker_address";
  TF_EXPECT_OK(RegisterWorker(address, state));
  std::shared_ptr<const Worker> worker;
  TF_EXPECT_OK(state.WorkerFromAddress(address, worker));
  EXPECT_EQ(worker->address, address);
}

TEST(DispatcherState, RegisterWorkerInFixedWorkerSet) {
  experimental::DispatcherConfig config;
  config.add_worker_addresses("/worker/task/0");
  config.add_worker_addresses("/worker/task/1");
  config.add_worker_addresses("/worker/task/2");

  DispatcherState state(config);
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/0:20000"));
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/1:20000"));
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/2:20000"));
  TF_EXPECT_OK(RegisterWorker("/worker/task/0:20000", state));
  TF_EXPECT_OK(RegisterWorker("/worker/task/1:20000", state));
  TF_EXPECT_OK(RegisterWorker("/worker/task/2:20000", state));

  std::shared_ptr<const Worker> worker;
  TF_EXPECT_OK(state.WorkerFromAddress("/worker/task/0:20000", worker));
  EXPECT_EQ(worker->address, "/worker/task/0:20000");
}

TEST(DispatcherState, RegisterInvalidWorkerInFixedWorkerSet) {
  experimental::DispatcherConfig config;
  config.add_worker_addresses("/worker/task/0");
  config.add_worker_addresses("/worker/task/1");
  config.add_worker_addresses("/worker/task/2");

  DispatcherState state(config);
  EXPECT_THAT(state.ValidateWorker("localhost:20000"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("The worker's address is not configured")));

  // Tests that `RegisterWorker` always returns OK, and ignores errors. This is
  // because the journal records are supposed to be valid. If there is an error,
  // it should be caught by `ValidateWorker` and not written to the journal.
  TF_EXPECT_OK(RegisterWorker("localhost:20000", state));
  std::shared_ptr<const Worker> worker;
  EXPECT_THAT(state.WorkerFromAddress("/worker/task/0:20000", worker),
              StatusIs(error::NOT_FOUND,
                       "Worker with address /worker/task/0:20000 not found."));
}

TEST(DispatcherState, ListWorkers) {
  DispatcherState state;
  std::string address_1 = "address_1";
  std::string address_2 = "address_2";
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, IsEmpty());
  }
  TF_EXPECT_OK(RegisterWorker(address_1, state));
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, SizeIs(1));
  }
  TF_EXPECT_OK(RegisterWorker(address_2, state));
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, SizeIs(2));
  }
}

TEST(DispatcherState, MissingWorker) {
  DispatcherState state;
  std::shared_ptr<const Worker> worker;
  Status s = state.WorkerFromAddress("test_worker_address", worker);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, UnknownUpdate) {
  DispatcherState state;
  Update update;
  Status s = state.Apply(update);
  EXPECT_EQ(s.code(), error::INTERNAL);
}

TEST(DispatcherState, JobName) {
  int64_t dataset_id = 10;
  DispatcherState state;
  int64_t job_id = state.NextAvailableJobId();
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  JobKey job_key(/*name=*/"test", /*iteration=*/1);
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, job_key, state));
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobByKey(job_key, job));
  EXPECT_EQ(state.NextAvailableJobId(), job_id + 1);
  EXPECT_EQ(job->dataset_id, dataset_id);
  EXPECT_EQ(job->job_id, job_id);
  EXPECT_FALSE(job->finished);
}

TEST(DispatcherState, NumConsumersJob) {
  int64_t dataset_id = 10;
  int64_t num_consumers = 8;
  DispatcherState state;
  int64_t job_id = state.NextAvailableJobId();
  TF_ASSERT_OK(RegisterDataset(dataset_id, state));
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->mutable_processing_mode_def()->set_sharding_policy(
      ProcessingModeDef::OFF);
  create_job->set_num_consumers(num_consumers);
  TF_ASSERT_OK(state.Apply(update));
  std::shared_ptr<const Job> job;
  TF_ASSERT_OK(state.JobFromId(job_id, job));
  EXPECT_EQ(job->num_consumers, num_consumers);
}

TEST(DispatcherState, CreateTask) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  int64_t task_id = state.NextAvailableTaskId();
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id, job_id, worker_address, state));
  EXPECT_EQ(state.NextAvailableTaskId(), task_id + 1);
  {
    std::shared_ptr<const Task> task;
    TF_EXPECT_OK(state.TaskFromId(task_id, task));
    EXPECT_EQ(task->job->job_id, job_id);
    EXPECT_EQ(task->task_id, task_id);
    EXPECT_EQ(task->worker_address, worker_address);
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(1, tasks.size());
  }
}

TEST(DispatcherState, CreateTasksForSameJob) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, job_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, job_id, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id, tasks));
    EXPECT_THAT(tasks, SizeIs(2));
  }
}

TEST(DispatcherState, CreateTasksForDifferentJobs) {
  int64_t job_id_1 = 3;
  int64_t job_id_2 = 4;
  int64_t dataset_id = 10;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id_1, dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id_2, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, job_id_1, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, job_id_2, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id_1, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id_2, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
}

TEST(DispatcherState, CreateTasksForSameWorker) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, job_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, job_id, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(2, tasks.size());
  }
}

TEST(DispatcherState, CreateTasksForDifferentWorkers) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address_1 = "test_worker_address_1";
  std::string worker_address_2 = "test_worker_address_2";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, job_id, worker_address_1, state));
  TF_EXPECT_OK(CreateTask(task_id_2, job_id, worker_address_2, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address_1, tasks));
    EXPECT_EQ(1, tasks.size());
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address_2, tasks));
    EXPECT_EQ(1, tasks.size());
  }
}

TEST(DispatcherState, GetTasksForWorkerEmpty) {
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterWorker(worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(0, tasks.size());
  }
}

TEST(DispatcherState, FinishTask) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t task_id = 4;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id, job_id, worker_address, state));
  TF_EXPECT_OK(FinishTask(task_id, state));
  std::shared_ptr<const Task> task;
  TF_EXPECT_OK(state.TaskFromId(task_id, task));
  EXPECT_TRUE(task->finished);
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, job));
  EXPECT_TRUE(job->finished);
}

TEST(DispatcherState, FinishMultiTaskJob) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t task_id_1 = 4;
  int64_t task_id_2 = 5;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, job_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, job_id, worker_address, state));

  TF_EXPECT_OK(FinishTask(task_id_1, state));
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobFromId(job_id, job));
    EXPECT_FALSE(job->finished);
  }

  TF_EXPECT_OK(FinishTask(task_id_2, state));
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobFromId(job_id, job));
    EXPECT_TRUE(job->finished);
  }
}

TEST(DispatcherState, AcquireJobClientId) {
  int64_t job_id = 3;
  int64_t job_client_id_1 = 1;
  int64_t job_client_id_2 = 2;
  int64_t dataset_id = 10;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id_1, state));
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobFromId(job_id, job));
    EXPECT_EQ(job->num_clients, 1);
    TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id_2, state));
    EXPECT_EQ(job->num_clients, 2);
  }
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobForJobClientId(job_client_id_1, job));
    EXPECT_EQ(job->job_id, job_id);
  }
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobForJobClientId(job_client_id_2, job));
    EXPECT_EQ(job->job_id, job_id);
  }
}

TEST(DispatcherState, ReleaseJobClientId) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t job_client_id = 6;
  int64_t release_time = 100;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id, state));
  TF_EXPECT_OK(ReleaseJobClientId(job_client_id, release_time, state));
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, job));
  EXPECT_EQ(job->num_clients, 0);
  Status s = state.JobForJobClientId(job_client_id, job);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, ListActiveClientsEmpty) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t job_client_id = 6;
  int64_t release_time = 100;
  DispatcherState state;
  EXPECT_THAT(state.ListActiveClientIds(), IsEmpty());
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id, state));
  TF_EXPECT_OK(ReleaseJobClientId(job_client_id, release_time, state));
  EXPECT_THAT(state.ListActiveClientIds(), IsEmpty());
}

TEST(DispatcherState, ListActiveClients) {
  int64_t job_id = 3;
  int64_t dataset_id = 10;
  int64_t job_client_id_1 = 6;
  int64_t job_client_id_2 = 7;
  int64_t job_client_id_3 = 8;
  int64_t release_time = 100;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id_1, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id_2, state));
  TF_EXPECT_OK(ReleaseJobClientId(job_client_id_2, release_time, state));
  TF_EXPECT_OK(AcquireJobClientId(job_id, job_client_id_3, state));
  EXPECT_THAT(state.ListActiveClientIds(), UnorderedElementsAre(6, 8));
}

}  // namespace data
}  // namespace tensorflow
