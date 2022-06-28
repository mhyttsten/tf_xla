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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc() {
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
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

StatusOr<std::unique_ptr<DataServiceWorkerClient>>
CreateDataServiceWorkerClient(const std::string& address,
                              const std::string& protocol,
                              const std::string& transfer_protocol) {
  auto client = absl::make_unique<DataServiceWorkerClient>(address, protocol,
                                                           transfer_protocol);
  TF_RETURN_IF_ERROR(client->Initialize());
  return client;
}

Status DataServiceWorkerClient::GetElement(const GetElementRequest& req,
                                           GetElementResult& result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/data/service/worker_client.cc", "DataServiceWorkerClient::GetElement");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  return client_->GetElement(req, result);
}

Status DataServiceWorkerClient::EnsureInitialized() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/data/service/worker_client.cc", "DataServiceWorkerClient::EnsureInitialized");

  mutex_lock l(mu_);
  if (client_) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(DataTransferClient::Build(
      GetDataTransferProtocol(), {protocol_, address_}, &client_));
  return Status::OK();
}

std::string DataServiceWorkerClient::GetDataTransferProtocol() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/data/service/worker_client.cc", "DataServiceWorkerClient::GetDataTransferProtocol");

  if (transfer_protocol_ == kGrpcTransferProtocol &&
      LocalWorkers::Get(address_) != nullptr) {
    return kLocalTransferProtocol;
  }
  return transfer_protocol_;
}

void DataServiceWorkerClient::TryCancel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/data/service/worker_client.cc", "DataServiceWorkerClient::TryCancel");
 client_->TryCancel(); }

class GrpcDataTransferClient : public DataTransferClient {
 public:
  GrpcDataTransferClient(std::shared_ptr<grpc::ChannelCredentials> credentials,
                         std::string address) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("address: \"" + address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/data/service/worker_client.cc", "GrpcDataTransferClient");

    VLOG(2) << "Create GrpcDataTransferClient for worker " << address << ".";
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(-1);
    auto channel = grpc::CreateCustomChannel(address, credentials, args);
    stub_ = WorkerService::NewStub(channel);
  }

  Status GetElement(const GetElementRequest& req,
                    GetElementResult& result) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/data/service/worker_client.cc", "GetElement");

    VLOG(3) << "GetElement for task " << req.task_id() << " from gRPC worker "
            << "server.";
    {
      mutex_lock l(mu_);
      if (cancelled_) {
        return errors::Cancelled("Client was cancelled.");
      }
    }
    grpc::ClientContext ctx;
    {
      mutex_lock l(mu_);
      active_contexts_.insert(&ctx);
    }
    GetElementResponse resp;
    grpc::Status s = stub_->GetElement(&ctx, req, &resp);
    result.end_of_sequence = resp.end_of_sequence();
    result.skip = resp.skip_task();
    switch (resp.element_case()) {
      case GetElementResponse::kCompressed: {
        Tensor tensor(DT_VARIANT, TensorShape{});
        tensor.scalar<Variant>()() = std::move(resp.compressed());
        result.components.push_back(tensor);
        break;
      }
      case GetElementResponse::kUncompressed:
        for (const auto& component : resp.uncompressed().components()) {
          result.components.emplace_back();
          if (!result.components.back().FromProto(component)) {
            return errors::Internal("Failed to parse tensor.");
          }
        }
        break;
      case GetElementResponse::ELEMENT_NOT_SET:
        break;
    }
    {
      mutex_lock l(mu_);
      active_contexts_.erase(&ctx);
    }
    if (!s.ok()) {
      return grpc_util::WrapError("Failed to get element", s);
    }
    return Status::OK();
  }

  void TryCancel() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_6(mht_6_v, 335, "", "./tensorflow/core/data/service/worker_client.cc", "TryCancel");

    VLOG(2) << "Cancel GrpcDataTransferClient.";
    mutex_lock l(mu_);
    cancelled_ = true;
    for (const auto& ctx : active_contexts_) {
      ctx->TryCancel();
    }
  }

 private:
  mutex mu_;
  std::unique_ptr<WorkerService::Stub> stub_;
  // Set of all currently active clients contexts. Used to support
  // cancellation.
  absl::flat_hash_set<::grpc::ClientContext*> active_contexts_
      TF_GUARDED_BY(mu_);
  // Indicates that the client has been cancelled, so no further requests should
  // be accepted.
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
};

class GrpcTransferClientRegistrar {
 public:
  GrpcTransferClientRegistrar() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_7(mht_7_v, 361, "", "./tensorflow/core/data/service/worker_client.cc", "GrpcTransferClientRegistrar");

    DataTransferClient::Register(
        kGrpcTransferProtocol, [](DataTransferClient::Config config,
                                  std::unique_ptr<DataTransferClient>* out) {
          std::shared_ptr<grpc::ChannelCredentials> credentials;
          TF_RETURN_IF_ERROR(CredentialsFactory::CreateClientCredentials(
              config.protocol, &credentials));
          *out = std::make_unique<GrpcDataTransferClient>(credentials,
                                                          config.address);
          return Status::OK();
        });
  }
};
static GrpcTransferClientRegistrar gprc_client_registrar;

class LocalDataTransferClient : public DataTransferClient {
 public:
  explicit LocalDataTransferClient(absl::string_view worker_address)
      : worker_address_(worker_address) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_8(mht_8_v, 383, "", "./tensorflow/core/data/service/worker_client.cc", "LocalDataTransferClient");

    VLOG(2) << "Create LocalDataTransferClient for worker " << worker_address_
            << ".";
  }

  Status GetElement(const GetElementRequest& req,
                    GetElementResult& result) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_9(mht_9_v, 392, "", "./tensorflow/core/data/service/worker_client.cc", "GetElement");

    VLOG(3) << "GetElement for task " << req.task_id() << " from local worker.";
    TF_RETURN_IF_ERROR(VerifyClientIsNotCancelled());
    TF_ASSIGN_OR_RETURN(std::shared_ptr<DataServiceWorkerImpl> worker,
                        GetWorker(req));
    return worker->GetElementResult(&req, &result);
  }

  void TryCancel() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_10(mht_10_v, 403, "", "./tensorflow/core/data/service/worker_client.cc", "TryCancel");

    VLOG(2) << "Cancel LocalDataTransferClient for worker " << worker_address_
            << ".";
    // Cancels incoming requests. Currently local reads assume the requests are
    // first-come-first-served. If we need to support coordinated reads, we need
    // to cancel in-flight requests since they may wait infinitely.
    mutex_lock l(mu_);
    cancelled_ = true;
  }

 private:
  Status VerifyClientIsNotCancelled() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (cancelled_) {
      return errors::Cancelled(absl::Substitute(
          "Client for worker $0 has been cancelled.", worker_address_));
    }
    return Status::OK();
  }

  StatusOr<std::shared_ptr<DataServiceWorkerImpl>> GetWorker(
      const GetElementRequest& req) const {
    std::shared_ptr<DataServiceWorkerImpl> worker =
        LocalWorkers::Get(worker_address_);
    if (!worker) {
      return errors::Cancelled(absl::Substitute(
          "Local worker at address $0 is no longer available; cancel request "
          "for task $1.",
          worker_address_, req.task_id()));
    }
    return worker;
  }

  const std::string worker_address_;

  mutex mu_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
};

class LocalTransferClientRegistrar {
 public:
  LocalTransferClientRegistrar() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_clientDTcc mht_11(mht_11_v, 447, "", "./tensorflow/core/data/service/worker_client.cc", "LocalTransferClientRegistrar");

    DataTransferClient::Register(
        kLocalTransferProtocol, [](DataTransferClient::Config config,
                                   std::unique_ptr<DataTransferClient>* out) {
          *out = absl::make_unique<LocalDataTransferClient>(config.address);
          return Status::OK();
        });
  }
};
static LocalTransferClientRegistrar local_client_registrar;

}  // namespace data
}  // namespace tensorflow
