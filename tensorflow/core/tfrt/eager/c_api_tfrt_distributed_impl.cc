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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc() {
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

#include "tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tfrt/distributed_runtime/distributed_context.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/distributed_init_helper.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/fabric_communicator.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/proto/cluster_config.pb.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/server_context.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/task_name_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

constexpr char kRemotePrefix[] = "remote_";

std::string GetTensorFlowDeviceType(string_view name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "GetTensorFlowDeviceType");

  int pos = name.find(kRemotePrefix);
  return absl::AsciiStrToUpper(
      pos == 0 ? name.substr(strlen(kRemotePrefix)).str() : name.str());
}

DistributedContextConfiguration ConvertServerDefToDistributedConfiguration(
    const tensorflow::ServerDef& server_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "ConvertServerDefToDistributedConfiguration");

  DistributedContextConfiguration dist_config;
  dist_config.set_job_name(server_def.job_name());
  dist_config.set_task_id(server_def.task_index());
  ClusterConfiguration* cluster_config = dist_config.mutable_cluster_config();
  // Currently take the first task in the first job as collective group leader.
  // TODO(haoyuzhang): Make this configurable from API by reading from
  // `config.experimental.collective_group_leader`.
  cluster_config->set_lead_task_name(TaskNameUtil::ConcatTaskName(
      server_def.cluster().job(0).name(), /*task_id=*/0));
  for (const auto& job_def : server_def.cluster().job()) {
    JobConfiguration* job_config = cluster_config->add_jobs();
    job_config->set_name(job_def.name());
    *job_config->mutable_tasks() = job_def.tasks();
  }
  return dist_config;
}

std::unique_ptr<ServerContext> CreateServer(
    const DistributedContextConfiguration& dist_config, HostContext* host_ctx) {
  const std::string& job_name = dist_config.job_name();
  const int task_id = dist_config.task_id();
  std::string server_address;
  for (const auto& job_config : dist_config.cluster_config().jobs()) {
    if (job_config.name() == job_name) {
      server_address = job_config.tasks().at(task_id);
      break;
    }
  }
  FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                server_address};
  ServerContextConfiguration server_config{fabric_config};
  return std::make_unique<ServerContext>(host_ctx, server_config);
}

}  // namespace

class DistributedManagerContextImpl
    : public DistributedManagerContextInterface {
 public:
  explicit DistributedManagerContextImpl(HostContext* host_context);

  tensorflow::Status SetOrUpdateServerDef(
      const tensorflow::ServerDef& server_def, bool reset_context,
      int keep_alive_secs) override;

  tensorflow::Status EnableCollectiveOps(
      const tensorflow::ServerDef& server_def) override;

  tensorflow::Status CheckRemoteAlive(const std::string& remote_task_name,
                                      bool* is_alive) override;

  tensorflow::CoordinationServiceAgent* GetCoordinationServiceAgent() override;

  void UpdateRequestContextBuilder(RequestContextBuilder* builder) override;
  void PopulateRemoteDevices(tensorflow::DeviceSet* dev_set) override;

 private:
  HostContext* host_context_;
  std::unique_ptr<tfrt::ServerContext> server_context_;
  AsyncValueRef<tfrt::DistributedContext> dist_context_;
  std::unique_ptr<tensorflow::StaticDeviceMgr> tf_devices_;
};

DistributedManagerContextImpl::DistributedManagerContextImpl(
    HostContext* host_context)
    : host_context_(host_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_2(mht_2_v, 289, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::DistributedManagerContextImpl");

  TaskNameUtil::SetUseReplicaInTaskName();
}

tensorflow::Status DistributedManagerContextImpl::SetOrUpdateServerDef(
    const tensorflow::ServerDef& server_def, bool reset_context,
    int keep_alive_secs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_3(mht_3_v, 298, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::SetOrUpdateServerDef");

#if defined(PLATFORM_GOOGLE)
  DistributedContextConfiguration dist_config =
      ConvertServerDefToDistributedConfiguration(server_def);
  server_context_ = CreateServer(dist_config, host_context_);

  // Create distributed contexts on current and remote tasks. Implemented as a
  // blocking call to be consistent with the behavior of current TF.
  const DistributedInitHelper* init_helper =
      server_context_->GetDistributedInitHelper();
  absl::Notification n;
  init_helper->InitializeSingleClientDistributedContext(
      std::move(dist_config),
      [&n, this](Expected<DistributedContext*> expected) mutable {
        if (!expected) tfrt::DieIfError(expected.takeError());
        const uint64_t cid = expected.get()->GetContextId();
        dist_context_ = server_context_->GetDistributedContextAsyncValue(cid);
        n.Notify();
      });
  n.WaitForNotification();

  auto device_refs =
      dist_context_->GetRemoteDeviceManager()->ListDevices<Device>();
  std::vector<std::unique_ptr<tensorflow::Device>> tf_devices;
  for (auto& device_ref : device_refs) {
    tensorflow::DeviceAttributes da;
    da.set_name(device_ref->name().str());
    da.set_device_type(GetTensorFlowDeviceType(device_ref->type().name()));
    // TF Devices created here might not have all of the attributes needed.
    // Currently, it is only used by Placer during TFRT Function creation.
    tf_devices.emplace_back(NewRemoteDevice(tensorflow::Env::Default(), da));
  }
  tf_devices_ =
      std::make_unique<tensorflow::StaticDeviceMgr>(std::move(tf_devices));
  return tensorflow::Status::OK();
#endif  // PLATFORM_GOOGLE
  return tensorflow::errors::Unimplemented(
      "SetOrUpdateServerDef in open source is not yet implemented.");
}

tensorflow::Status DistributedManagerContextImpl::EnableCollectiveOps(
    const tensorflow::ServerDef& server_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_4(mht_4_v, 342, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::EnableCollectiveOps");

#if defined(PLATFORM_GOOGLE)
  DistributedContextConfiguration dist_config =
      ConvertServerDefToDistributedConfiguration(server_def);
  server_context_ = CreateServer(dist_config, host_context_);

  DistributedInitHelper* init_helper =
      server_context_->GetDistributedInitHelper();
  absl::Notification n;
  init_helper->InitializeMultiClientDistributedContext(
      std::move(dist_config),
      [&n, this](Expected<DistributedContext*> expected) mutable {
        if (!expected) tfrt::DieIfError(expected.takeError());
        const uint64_t cid = expected.get()->GetContextId();
        dist_context_ = server_context_->GetDistributedContextAsyncValue(cid);
        n.Notify();
      });
  n.WaitForNotification();

  return tensorflow::Status::OK();
#endif  // PLATFORM_GOOGLE
  return tensorflow::errors::Unimplemented(
      "EnableCollectiveOps in open source is not yet implemented.");
}

tensorflow::Status DistributedManagerContextImpl::CheckRemoteAlive(
    const std::string& remote_task_name, bool* is_alive) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("remote_task_name: \"" + remote_task_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_5(mht_5_v, 372, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::CheckRemoteAlive");

  return tensorflow::errors::Unimplemented(
      "CheckRemoteAlive in TFRT is not yet implemented.");
}

tensorflow::CoordinationServiceAgent*
DistributedManagerContextImpl::GetCoordinationServiceAgent() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_6(mht_6_v, 381, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::GetCoordinationServiceAgent");

  TFRT_LOG(FATAL) << "Coordination service in TFRT is not yet enabled.";
  return nullptr;
}

void DistributedManagerContextImpl::UpdateRequestContextBuilder(
    RequestContextBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_7(mht_7_v, 390, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::UpdateRequestContextBuilder");

  builder->context_data().insert(dist_context_.CopyRef());
}

void DistributedManagerContextImpl::PopulateRemoteDevices(
    tensorflow::DeviceSet* dev_set) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrt_distributed_implDTcc mht_8(mht_8_v, 398, "", "./tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.cc", "DistributedManagerContextImpl::PopulateRemoteDevices");

  if (tf_devices_ == nullptr) {
    return;
  }
  for (auto& device : tf_devices_->ListDevices()) {
    dev_set->AddDevice(device);
  }
}

std::unique_ptr<DistributedManagerContextInterface>
CreateDistributedManagerContext(HostContext* host_context) {
  return std::make_unique<DistributedManagerContextImpl>(host_context);
}

}  // namespace tf
}  // namespace tfrt
