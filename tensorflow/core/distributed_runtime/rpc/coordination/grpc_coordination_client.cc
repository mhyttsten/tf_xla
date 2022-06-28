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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"

#include <string>
#include <utility>

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {
namespace {

class GrpcCoordinationClientThread {
 public:
  GrpcCoordinationClientThread() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GrpcCoordinationClientThread");

    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "coordination_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcCoordinationClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcCoordinationClientThread blocking for next tag";
          }
          VLOG(4) << "GrpcCoordinationClientThread exiting";
        }));
  }

  ~GrpcCoordinationClientThread() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "~GrpcCoordinationClientThread");

    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "completion_queue");
 return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class GrpcCoordinationClient : public CoordinationClient {
 public:
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         ::grpc::CompletionQueue* cq, const std::string& target)
      : stub_(channel), cq_(cq), target_(target) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GrpcCoordinationClient");
}
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         const std::string& target)
      : stub_(channel), target_(target) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GrpcCoordinationClient");

    client_thread_ = std::make_unique<GrpcCoordinationClientThread>();
    cq_ = client_thread_->completion_queue();
  }
  ~GrpcCoordinationClient() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_5(mht_5_v, 258, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "~GrpcCoordinationClient");
}

  void RegisterTaskAsync(CallOptions* call_opts,
                         const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_6(mht_6_v, 266, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "RegisterTaskAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/RegisterTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/false,
        &target_);
  }

  void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                            WaitForAllTasksResponse* response,
                            StatusCallback done) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "WaitForAllTasksAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/WaitForAllTasks",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ShutdownTaskAsync(CallOptions* call_opts,
                         const ShutdownTaskRequest* request,
                         ShutdownTaskResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_8(mht_8_v, 293, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "ShutdownTaskAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ShutdownTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ResetTaskAsync(const ResetTaskRequest* request,
                      ResetTaskResponse* response,
                      StatusCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "ResetTaskAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ResetTask", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void HeartbeatAsync(CallOptions* call_opts, const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_10(mht_10_v, 319, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "HeartbeatAsync");

    // Different from other RPCs which do not retry by default, the Heartbeat
    // RPC should retry automatically to tolerate transient network issues.
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Heartbeat", *request,
        response, std::move(done), call_opts, /*threadpool=*/nullptr,
        /*max_retries=*/3,
        /*fail_fast=*/true, &target_);
  }

  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_11(mht_11_v, 335, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "ReportErrorToTaskAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ReportErrorToTask",
        *request, response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ReportErrorToServiceAsync(const ReportErrorToServiceRequest* request,
                                 ReportErrorToServiceResponse* response,
                                 StatusCallback done) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_12(mht_12_v, 348, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "ReportErrorToServiceAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ReportErrorToService",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           StatusCallback done) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_13(mht_13_v, 361, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "InsertKeyValueAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/InsertKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueAsync(const GetKeyValueRequest* request,
                        GetKeyValueResponse* response,
                        StatusCallback done) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_14(mht_14_v, 374, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GetKeyValueAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                           DeleteKeyValueResponse* response,
                           StatusCallback done) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_15(mht_15_v, 387, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "DeleteKeyValueAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/DeleteKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void BarrierAsync(const BarrierRequest* request, BarrierResponse* response,
                    StatusCallback done) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_16(mht_16_v, 399, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "BarrierAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Barrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void CancelBarrierAsync(const CancelBarrierRequest* request,
                          CancelBarrierResponse* response,
                          StatusCallback done) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_17(mht_17_v, 412, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "CancelBarrierAsync");

    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/CancelBarrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  const string target_;
  std::unique_ptr<GrpcCoordinationClientThread> client_thread_;
};

class GrpcCoordinationClientCache : public CoordinationClientCache {
 public:
  explicit GrpcCoordinationClientCache(
      std::shared_ptr<GrpcChannelCache> channel_cache)
      : next_round_robin_assignment_(0),
        channel_cache_(channel_cache),
        threads_(4) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_18(mht_18_v, 436, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GrpcCoordinationClientCache");
}

  ~GrpcCoordinationClientCache() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_19(mht_19_v, 441, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "~GrpcCoordinationClientCache");
}

  CoordinationClient* GetClient(const string& target) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_20(mht_20_v, 447, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "GetClient");

    mutex_lock l(clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (channel == nullptr) {
        VLOG(2) << "Coordination client for target " << target << " not found.";
      }
      int assigned_index = AssignClientToThread(target);
      auto coord_client = std::make_unique<GrpcCoordinationClient>(
          channel, threads_[assigned_index].completion_queue(), target);
      it = clients_.emplace(target, std::move(coord_client)).first;
    }
    return it->second.get();
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const string& target) override {
    SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
    if (channel == nullptr) {
      VLOG(2) << "Coordination client for target " << target << " not found.";
    }
    return std::make_unique<GrpcCoordinationClient>(channel, target);
  }

 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const string& target) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_21(mht_21_v, 482, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "AssignClientToThread");

    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  std::shared_ptr<GrpcChannelCache> channel_cache_;
  mutable mutex clients_mu_;
  std::unordered_map<std::string, std::unique_ptr<CoordinationClient>> clients_
      TF_GUARDED_BY(clients_mu_);
  std::vector<GrpcCoordinationClientThread> threads_;
};

}  // namespace

CoordinationClientCache* NewGrpcCoordinationClientCache(
    std::shared_ptr<GrpcChannelCache> channel_cache) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_22(mht_22_v, 509, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "NewGrpcCoordinationClientCache");

  return new GrpcCoordinationClientCache(channel_cache);
}

CoordinationClient* NewGrpcCoordinationClient(
    std::shared_ptr<::grpc::Channel> channel) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPScoordinationPSgrpc_coordination_clientDTcc mht_23(mht_23_v, 517, "", "./tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.cc", "NewGrpcCoordinationClient");

  // TODO(hanyangtay): Pass in the logical task name for better logging.
  return new GrpcCoordinationClient(
      channel, /*target=*/"unknown_target_for_coordination_leader");
}

}  // namespace tensorflow
