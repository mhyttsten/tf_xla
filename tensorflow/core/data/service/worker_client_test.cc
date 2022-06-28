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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc() {
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
#include "tensorflow/core/data/service/worker_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::RangeSquareDataset;
using ::tensorflow::testing::StatusIs;
using ::testing::MatchesRegex;

constexpr const char kProtocol[] = "grpc";

class WorkerClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/data/service/worker_client_test.cc", "SetUp");

    test_cluster_ = absl::make_unique<TestCluster>(/*num_workers=*/1);
    TF_ASSERT_OK(test_cluster_->Initialize());
    dispatcher_client_ = absl::make_unique<DataServiceDispatcherClient>(
        test_cluster_->DispatcherAddress(), kProtocol);
  }

  // Creates a dataset and returns the dataset ID.
  StatusOr<int64_t> RegisterDataset(const int64_t range) {
    const auto dataset_def = RangeSquareDataset(range);
    int64_t dataset_id = 0;
    TF_RETURN_IF_ERROR(dispatcher_client_->RegisterDataset(
        dataset_def, DataServiceMetadata(), dataset_id));
    return dataset_id;
  }

  // Creates a job and returns the job client ID.
  StatusOr<int64_t> CreateJob(const int64_t dataset_id) {
    ProcessingModeDef processing_mode;
    processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
    int64_t job_client_id = 0;
    TF_RETURN_IF_ERROR(dispatcher_client_->GetOrCreateJob(
        dataset_id, processing_mode, /*job_key=*/absl::nullopt,
        /*num_consumers=*/absl::nullopt, TARGET_WORKERS_AUTO, job_client_id));
    return job_client_id;
  }

  // Gets the task for job `job_client_id`.
  StatusOr<int64_t> GetTaskToRead(const int64_t job_client_id) {
    ClientHeartbeatRequest request;
    ClientHeartbeatResponse response;
    request.set_job_client_id(job_client_id);
    TF_RETURN_IF_ERROR(dispatcher_client_->ClientHeartbeat(request, response));
    if (response.task_info().empty()) {
      return errors::NotFound(
          absl::Substitute("No task found for job $0.", job_client_id));
    }
    return response.task_info(0).task_id();
  }

  StatusOr<std::unique_ptr<DataServiceWorkerClient>> GetWorkerClient(
      const std::string& data_transfer_protocol) {
    return CreateDataServiceWorkerClient(
        GetWorkerAddress(), /*protocol=*/kProtocol, data_transfer_protocol);
  }

  StatusOr<GetElementResult> GetElement(DataServiceWorkerClient& client,
                                        const int64_t task_id) {
    GetElementRequest request;
    GetElementResult result;
    request.set_task_id(task_id);
    TF_RETURN_IF_ERROR(client.GetElement(request, result));
    return result;
  }

  std::string GetDispatcherAddress() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc mht_1(mht_1_v, 285, "", "./tensorflow/core/data/service/worker_client_test.cc", "GetDispatcherAddress");

    return test_cluster_->DispatcherAddress();
  }

  std::string GetWorkerAddress() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_client_testDTcc mht_2(mht_2_v, 292, "", "./tensorflow/core/data/service/worker_client_test.cc", "GetWorkerAddress");

    return test_cluster_->WorkerAddress(0);
  }

  std::unique_ptr<TestCluster> test_cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
};

TEST_F(WorkerClientTest, LocalRead) {
  const int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(const int64_t dataset_id, RegisterDataset(range));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t job_client_id, CreateJob(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id, GetTaskToRead(job_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));
  for (int64_t i = 0; i < range; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                            GetElement(*client, task_id));
    test::ExpectEqual(result.components[0], Tensor(int64_t{i * i}));
    EXPECT_FALSE(result.end_of_sequence);
  }

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, LocalReadEmptyDataset) {
  TF_ASSERT_OK_AND_ASSIGN(const int64_t dataset_id,
                          RegisterDataset(/*range=*/0));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t job_client_id, CreateJob(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id, GetTaskToRead(job_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));
  TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                          GetElement(*client, task_id));
  EXPECT_TRUE(result.end_of_sequence);

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, GrpcRead) {
  const int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(const int64_t dataset_id, RegisterDataset(range));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t job_client_id, CreateJob(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id, GetTaskToRead(job_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kGrpcTransferProtocol));
  for (int64_t i = 0; i < range; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                            GetElement(*client, task_id));
    test::ExpectEqual(result.components[0], Tensor(int64_t{i * i}));
    EXPECT_FALSE(result.end_of_sequence);
  }

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, LocalServerShutsDown) {
  TF_ASSERT_OK_AND_ASSIGN(const int64_t dataset_id,
                          RegisterDataset(/*range=*/5));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t job_client_id, CreateJob(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id, GetTaskToRead(job_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));

  // Stopping a worker causes local reads to return Cancelled status.
  test_cluster_->StopWorkers();
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, CancelClient) {
  TF_ASSERT_OK_AND_ASSIGN(const int64_t dataset_id,
                          RegisterDataset(/*range=*/5));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t job_client_id, CreateJob(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id, GetTaskToRead(job_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));

  client->TryCancel();
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Client for worker.*has been cancelled.")));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
