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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc() {
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
#include "tensorflow/core/data/service/dispatcher_client.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

StatusOr<WorkerHeartbeatResponse> DataServiceDispatcherClient::WorkerHeartbeat(
    const WorkerHeartbeatRequest& request) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::WorkerHeartbeat");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  WorkerHeartbeatResponse response;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerHeartbeat(&client_ctx, request, &response);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to perform worker heartbeat", status);
  }
  return response;
}

Status DataServiceDispatcherClient::WorkerUpdate(
    const std::string& worker_address,
    std::vector<TaskProgress>& task_progress) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("worker_address: \"" + worker_address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::WorkerUpdate");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  WorkerUpdateRequest req;
  req.set_worker_address(worker_address);
  for (const auto& update : task_progress) {
    *(req.add_updates()) = update;
  }
  WorkerUpdateResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerUpdate(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to send worker update", status);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::GetDatasetDef(int64_t dataset_id,
                                                  DatasetDef& dataset_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetDatasetDef");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDatasetDefRequest req;
  req.set_dataset_id(dataset_id);
  GetDatasetDefResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetDatasetDef(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get dataset def", status);
  }
  dataset_def = resp.dataset_def();
  return Status::OK();
}

Status DataServiceDispatcherClient::GetSplit(int64_t job_id, int64_t iteration,
                                             int64_t split_provider_index,
                                             Tensor& split,
                                             bool& end_of_splits) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_3(mht_3_v, 274, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetSplit");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetSplitRequest req;
  req.set_job_id(job_id);
  req.set_iteration(iteration);
  req.set_split_provider_index(split_provider_index);
  GetSplitResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetSplit(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get split", status);
  }
  end_of_splits = resp.end_of_splits();
  if (!end_of_splits) {
    if (!split.FromProto(resp.split())) {
      return errors::Internal("Failed to parse split tensor proto");
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::RegisterDataset(
    const DatasetDef& dataset, const DataServiceMetadata& metadata,
    int64_t& dataset_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::RegisterDataset");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset() = dataset;
  *req.mutable_metadata() = metadata;

  GetOrRegisterDatasetResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrRegisterDataset(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to register dataset", status);
  }
  dataset_id = resp.dataset_id();
  return Status::OK();
}

Status DataServiceDispatcherClient::GetOrCreateJob(
    int64_t dataset_id, const ProcessingModeDef& processing_mode,
    const absl::optional<JobKeyDef>& job_key,
    absl::optional<int64_t> num_consumers, TargetWorkers target_workers,
    int64_t& job_client_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_5(mht_5_v, 323, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetOrCreateJob");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrCreateJobRequest req;
  req.set_dataset_id(dataset_id);
  *req.mutable_processing_mode_def() = processing_mode;
  if (job_key.has_value()) {
    *req.mutable_job_key() = job_key.value();
  }
  if (num_consumers.has_value()) {
    req.set_num_consumers(num_consumers.value());
  }
  req.set_target_workers(target_workers);
  GetOrCreateJobResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrCreateJob(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get or create job for dataset with id ",
                     dataset_id),
        status);
  }
  job_client_id = resp.job_client_id();
  return Status::OK();
}

Status DataServiceDispatcherClient::ReleaseJobClient(int64_t job_client_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_6(mht_6_v, 351, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::ReleaseJobClient");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  ReleaseJobClientRequest req;
  req.set_job_client_id(job_client_id);
  ReleaseJobClientResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->ReleaseJobClient(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to release job client with id ", job_client_id),
        status);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::MaybeRemoveTask(int64_t task_id,
                                                    int64_t consumer_index,
                                                    int64_t round,
                                                    bool& removed) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_7(mht_7_v, 372, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::MaybeRemoveTask");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  MaybeRemoveTaskRequest req;
  req.set_task_id(task_id);
  req.set_consumer_index(consumer_index);
  req.set_round(round);
  MaybeRemoveTaskResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->MaybeRemoveTask(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to call MaybeRemoveTask", status);
  }
  removed = resp.removed();
  return Status::OK();
}

Status DataServiceDispatcherClient::ClientHeartbeat(
    ClientHeartbeatRequest& req, ClientHeartbeatResponse& resp) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_8(mht_8_v, 392, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::ClientHeartbeat");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  grpc::ClientContext ctx;
  grpc::Status s = stub_->ClientHeartbeat(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::GetWorkers(
    std::vector<WorkerInfo>& workers) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_9(mht_9_v, 406, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetWorkers");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetWorkersRequest req;
  GetWorkersResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetWorkers(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  workers.clear();
  for (auto& worker : resp.workers()) {
    workers.push_back(worker);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::GetDataServiceMetadata(
    int64_t dataset_id, DataServiceMetadata& metadata) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_10(mht_10_v, 426, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetDataServiceMetadata");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDataServiceMetadataRequest req;
  req.set_dataset_id(dataset_id);
  GetDataServiceMetadataResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetDataServiceMetadata(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get data service metadata", s);
  }
  metadata = resp.metadata();
  return Status::OK();
}

Status DataServiceDispatcherClient::GetDataServiceConfig(
    DataServiceConfig& config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_11(mht_11_v, 444, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::GetDataServiceConfig");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDataServiceConfigRequest request;
  GetDataServiceConfigResponse response;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetDataServiceConfig(&ctx, request, &response);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get data service config", s);
  }
  config = response.config();
  return Status::OK();
}

Status DataServiceDispatcherClient::EnsureInitialized() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTcc mht_12(mht_12_v, 460, "", "./tensorflow/core/data/service/dispatcher_client.cc", "DataServiceDispatcherClient::EnsureInitialized");

  mutex_lock l(mu_);
  if (stub_) {
    return Status::OK();
  }
  std::shared_ptr<grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
  auto channel = grpc::CreateCustomChannel(address_, credentials, args);
  stub_ = DispatcherService::NewStub(channel);
  GetVersionRequest req;
  GetVersionResponse resp;
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&] {
        grpc::ClientContext ctx;
        grpc::Status s = stub_->GetVersion(&ctx, req, &resp);
        if (!s.ok()) {
          return grpc_util::WrapError(
              absl::StrCat("Failed to get dispatcher version from dispatcher "
                           "running at ",
                           address_),
              s);
        }
        return Status::OK();
      },
      "check service version",
      /*deadline_micros=*/kint64max));
  if (resp.version() != kDataServiceVersion) {
    return errors::FailedPrecondition(
        "Version mismatch with tf.data service server. The server is running "
        "version ",
        resp.version(), ", while the client is running version ",
        kDataServiceVersion,
        ". Please ensure that the client and server side are running the "
        "same version of TensorFlow.");
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
