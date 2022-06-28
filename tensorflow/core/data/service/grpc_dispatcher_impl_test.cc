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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc() {
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
#include "tensorflow/core/data/service/grpc_dispatcher_impl.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::grpc::Channel;
using ::grpc::ChannelArguments;
using ::grpc::ChannelCredentials;
using ::grpc::ClientContext;
using ::tensorflow::data::testing::RangeSquareDataset;

constexpr const char kHostAddress[] = "localhost";
constexpr const char kProtocol[] = "grpc";

class GrpcDispatcherImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/data/service/grpc_dispatcher_impl_test.cc", "SetUp");

    TF_ASSERT_OK(SetUpDispatcherServer());
    TF_ASSERT_OK(SetUpDispatcherClientStub());
  }

  Status SetUpDispatcherServer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/data/service/grpc_dispatcher_impl_test.cc", "SetUpDispatcherServer");

    experimental::DispatcherConfig config;
    config.set_protocol(kProtocol);
    TF_RETURN_IF_ERROR(NewDispatchServer(config, dispatcher_server_));
    return dispatcher_server_->Start();
  }

  Status SetUpDispatcherClientStub() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/data/service/grpc_dispatcher_impl_test.cc", "SetUpDispatcherClientStub");

    std::shared_ptr<ChannelCredentials> credentials;
    TF_RETURN_IF_ERROR(
        CredentialsFactory::CreateClientCredentials(kProtocol, &credentials));
    ChannelArguments args;
    args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    std::shared_ptr<Channel> channel =
        ::grpc::CreateCustomChannel(GetDispatcherAddress(), credentials, args);
    dispatcher_client_stub_ = DispatcherService::NewStub(channel);
    return Status::OK();
  }

  std::string GetDispatcherAddress() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSgrpc_dispatcher_impl_testDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/data/service/grpc_dispatcher_impl_test.cc", "GetDispatcherAddress");

    return absl::StrCat(kHostAddress, ":", dispatcher_server_->BoundPort());
  }

  StatusOr<GetOrRegisterDatasetResponse> RegisterDataset() {
    GetOrRegisterDatasetRequest request;
    GetOrRegisterDatasetResponse response;
    *request.mutable_dataset() = RangeSquareDataset(/*range=*/10);
    ClientContext context;
    TF_RETURN_IF_ERROR(
        FromGrpcStatus(dispatcher_client_stub_->GetOrRegisterDataset(
            &context, request, &response)));
    return response;
  }

  StatusOr<GetOrCreateJobResponse> CreateJob(const int64_t dataset_id) {
    GetOrCreateJobRequest request;
    GetOrCreateJobResponse response;
    request.set_dataset_id(dataset_id);
    request.mutable_processing_mode_def()->set_sharding_policy(
        ProcessingModeDef::OFF);
    ClientContext context;
    TF_RETURN_IF_ERROR(FromGrpcStatus(
        dispatcher_client_stub_->GetOrCreateJob(&context, request, &response)));
    return response;
  }

  StatusOr<WorkerHeartbeatResponse> WorkerHeartbeat() {
    WorkerHeartbeatRequest request;
    WorkerHeartbeatResponse response;
    request.set_worker_address(kHostAddress);
    request.set_transfer_address(kHostAddress);
    ClientContext client_ctx;
    TF_RETURN_IF_ERROR(FromGrpcStatus(dispatcher_client_stub_->WorkerHeartbeat(
        &client_ctx, request, &response)));
    return response;
  }

  StatusOr<ClientHeartbeatResponse> ClientHeartbeat(
      const int64_t job_client_id) {
    ClientHeartbeatRequest request;
    ClientHeartbeatResponse response;
    request.set_job_client_id(job_client_id);
    ClientContext client_ctx;
    TF_RETURN_IF_ERROR(FromGrpcStatus(dispatcher_client_stub_->ClientHeartbeat(
        &client_ctx, request, &response)));
    return response;
  }

  std::unique_ptr<DispatchGrpcDataServer> dispatcher_server_;
  std::unique_ptr<DispatcherService::Stub> dispatcher_client_stub_;
};

TEST_F(GrpcDispatcherImplTest, WorkerHeartbeat) {
  TF_ASSERT_OK_AND_ASSIGN(GetOrRegisterDatasetResponse dataset_response,
                          RegisterDataset());
  TF_ASSERT_OK(CreateJob(dataset_response.dataset_id()).status());
  TF_ASSERT_OK_AND_ASSIGN(WorkerHeartbeatResponse worker_response,
                          WorkerHeartbeat());
  ASSERT_EQ(worker_response.new_tasks().size(), 1);
  EXPECT_EQ(worker_response.new_tasks(0).dataset_id(),
            dataset_response.dataset_id());
}

TEST_F(GrpcDispatcherImplTest, ClientHeartbeat) {
  TF_ASSERT_OK_AND_ASSIGN(GetOrRegisterDatasetResponse dataset_response,
                          RegisterDataset());
  TF_ASSERT_OK_AND_ASSIGN(GetOrCreateJobResponse job_response,
                          CreateJob(dataset_response.dataset_id()));
  TF_ASSERT_OK(WorkerHeartbeat().status());
  TF_ASSERT_OK_AND_ASSIGN(ClientHeartbeatResponse client_response,
                          ClientHeartbeat(job_response.job_client_id()));
  ASSERT_EQ(client_response.task_info().size(), 1);
  EXPECT_EQ(client_response.task_info(0).worker_address(), kHostAddress);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
