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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc() {
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

#include "tensorflow/core/data/service/server_lib.h"

#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_dispatcher_impl.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/grpc_worker_impl.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

namespace {
constexpr char kPortPlaceholder[] = "%port%";
}

GrpcDataServerBase::GrpcDataServerBase(int port, const std::string& protocol,
                                       const std::string server_type)
    : requested_port_(port),
      protocol_(protocol),
      server_type_(server_type),
      bound_port_(port) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("protocol: \"" + protocol + "\"");
   mht_0_v.push_back("server_type: \"" + server_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::GrpcDataServerBase");
}

Status GrpcDataServerBase::Start() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::Start");

  if (stopped_) {
    return errors::FailedPrecondition(
        "Server cannot be started after it has been stopped.");
  }
  if (started_) {
    return Status::OK();
  }
  ::grpc::ServerBuilder builder;
  std::shared_ptr<::grpc::ServerCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateServerCredentials(protocol_, &credentials));
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                           credentials, &bound_port_);
  builder.SetMaxReceiveMessageSize(-1);

  AddDataServiceToBuilder(builder);
  AddProfilerServiceToBuilder(builder);
  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Internal("Could not start gRPC server");
  }

  TF_RETURN_IF_ERROR(StartServiceInternal());

  started_ = true;
  LOG(INFO) << "Started tf.data " << server_type_
            << " running at 0.0.0.0:" << BoundPort();
  return Status::OK();
}

void GrpcDataServerBase::Stop() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::Stop");

  if (stopped_) {
    return;
  }
  if (server_) {
    StopServiceInternal();
    server_->Shutdown();
    LOG(INFO) << "Shut down " << server_type_ << " server running at port "
              << BoundPort();
  }
  stopped_ = true;
}

void GrpcDataServerBase::Join() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::Join");
 server_->Wait(); }

int GrpcDataServerBase::BoundPort() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::BoundPort");
 return bound_port(); }

void GrpcDataServerBase::AddProfilerServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_5(mht_5_v, 273, "", "./tensorflow/core/data/service/server_lib.cc", "GrpcDataServerBase::AddProfilerServiceToBuilder");

  profiler_service_ = profiler::CreateProfilerService();
  builder.RegisterService(profiler_service_.get());
}

DispatchGrpcDataServer::DispatchGrpcDataServer(
    const experimental::DispatcherConfig& config)
    : GrpcDataServerBase(config.port(), config.protocol(), "DispatchServer"),
      config_(config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::DispatchGrpcDataServer");
}

DispatchGrpcDataServer::~DispatchGrpcDataServer() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_7(mht_7_v, 289, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::~DispatchGrpcDataServer");
 delete service_; }

void DispatchGrpcDataServer::AddDataServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_8(mht_8_v, 295, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::AddDataServiceToBuilder");

  service_ = absl::make_unique<GrpcDispatcherImpl>(config_, builder).release();
}

Status DispatchGrpcDataServer::StartServiceInternal() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_9(mht_9_v, 302, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::StartServiceInternal");

  return service_->Start();
}

Status DispatchGrpcDataServer::NumWorkers(int* num_workers) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_10(mht_10_v, 309, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::NumWorkers");

  GetWorkersRequest req;
  GetWorkersResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetWorkers(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  *num_workers = resp.workers_size();
  return Status::OK();
}

size_t DispatchGrpcDataServer::NumActiveJobs() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_11(mht_11_v, 324, "", "./tensorflow/core/data/service/server_lib.cc", "DispatchGrpcDataServer::NumActiveJobs");

  return service_->NumActiveJobs();
}

WorkerGrpcDataServer::WorkerGrpcDataServer(
    const experimental::WorkerConfig& config)
    : GrpcDataServerBase(config.port(), config.protocol(), "WorkerServer"),
      config_(config) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_12(mht_12_v, 334, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::WorkerGrpcDataServer");
}

WorkerGrpcDataServer::~WorkerGrpcDataServer() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_13(mht_13_v, 339, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::~WorkerGrpcDataServer");
 delete service_; }

void WorkerGrpcDataServer::AddDataServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_14(mht_14_v, 345, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::AddDataServiceToBuilder");

  service_ = absl::make_unique<GrpcWorkerImpl>(config_, builder).release();
}

Status WorkerGrpcDataServer::StartServiceInternal() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_15(mht_15_v, 352, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::StartServiceInternal");

  std::string base_address = config_.worker_address();
  if (base_address.empty()) {
    base_address = absl::StrCat("localhost:", kPortPlaceholder);
  }
  std::string worker_address = str_util::StringReplace(
      base_address, kPortPlaceholder, absl::StrCat(bound_port()),
      /*replace_all=*/false);
  std::string transfer_address = worker_address;
  std::string transfer_protocol = config_.data_transfer_protocol();
  if (!transfer_protocol.empty() && transfer_protocol != "grpc") {
    TF_RETURN_IF_ERROR(DataTransferServer::Build(
        transfer_protocol, service_->get_element_getter(), &transfer_server_));
    TF_RETURN_IF_ERROR(transfer_server_->Start());
    LOG(INFO) << "Data transfer server started at 0.0.0.0:"
              << transfer_server_->get_port();
    transfer_address = str_util::StringReplace(
        config_.data_transfer_address(), kPortPlaceholder,
        absl::StrCat(transfer_server_->get_port()),
        /*replace_all=*/false);
  }
  TF_RETURN_IF_ERROR(service_->Start(worker_address, transfer_address));
  return Status::OK();
}

void WorkerGrpcDataServer::StopServiceInternal() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_16(mht_16_v, 380, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::StopServiceInternal");
 service_->Stop(); }

Status WorkerGrpcDataServer::NumTasks(int* num_tasks) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_17(mht_17_v, 385, "", "./tensorflow/core/data/service/server_lib.cc", "WorkerGrpcDataServer::NumTasks");

  GetWorkerTasksRequest req;
  GetWorkerTasksResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetWorkerTasks(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  *num_tasks = resp.tasks_size();
  return Status::OK();
}

Status NewDispatchServer(const experimental::DispatcherConfig& config,
                         std::unique_ptr<DispatchGrpcDataServer>& out_server) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_18(mht_18_v, 401, "", "./tensorflow/core/data/service/server_lib.cc", "NewDispatchServer");

  out_server = absl::make_unique<DispatchGrpcDataServer>(config);
  return Status::OK();
}

Status NewWorkerServer(const experimental::WorkerConfig& config,
                       std::unique_ptr<WorkerGrpcDataServer>& out_server) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSserver_libDTcc mht_19(mht_19_v, 410, "", "./tensorflow/core/data/service/server_lib.cc", "NewWorkerServer");

  out_server = absl::make_unique<WorkerGrpcDataServer>(config);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
