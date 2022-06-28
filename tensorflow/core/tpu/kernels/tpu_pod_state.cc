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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_pod_state.h"

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/tpu/tpu_api.h"

#if defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#else
#include "tensorflow/core/tpu/kernels/tpu_util.h"  // copybara"
#endif

namespace tensorflow {
const char kTpuPodStateResourceName[] = "tpu_pod_state";

namespace {

// Attempt to delete resource_name from resource_manager's default_container.
// Returns OK if the deletion succeeded, or if the resource was not found. Else
// return the deletion error.
template <class ResourceT>
Status DeleteIfExists(ResourceMgr* resource_manager,
                      const char* resource_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("resource_name: \"" + (resource_name == nullptr ? std::string("nullptr") : std::string((char*)resource_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "DeleteIfExists");

  VLOG(1) << "Removing resource " << resource_name << " if it exists";
  Status status = resource_manager->Delete<ResourceT>(
      resource_manager->default_container(), resource_name);
  if (status.ok()) {
    VLOG(1) << "Removed existing resource " << resource_name;
    return Status::OK();
  }
  if (status.code() == error::NOT_FOUND) {
    VLOG(1) << "No resource " << resource_name << " to remove";
    return Status::OK();
  }
  VLOG(1) << "Error removing resource " << resource_name << " : " << status;
  return status;
}

xla::StatusOr<std::unique_ptr<TpuCompilationCacheService>>
ConstructCacheService(ResourceMgr* rmgr, int serving_port,
                      tpu::TpuCompilationCacheInterface* compilation_cache) {
  xla::StatusOr<std::unique_ptr<::grpc::ServerBuilder>> server_builder;
#if defined(LIBTPU_ON_GCE)
  server_builder = tpu::CreateServerBuilder(serving_port);
#else
  server_builder = tpu::CreateServerBuilderGoogle(serving_port);
#endif
  TF_RETURN_IF_ERROR(server_builder.status());

  auto cache_service = absl::make_unique<TpuCompilationCacheService>(
      server_builder.ValueOrDie().get(), compilation_cache);
  cache_service->SetMemoryQuota(1ul << 31);  // 2GB
  cache_service->Start();
  return cache_service;
}
}  // namespace

Status GetServerAddressAndPort(std::string* server_address, int* serving_port) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "GetServerAddressAndPort");

  TF_Status* status = TF_NewStatus();
  char* server_address_output = nullptr;
  auto cleanup = absl::MakeCleanup([&status, &server_address_output]() {
    TF_DeleteStatus(status);
    tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(server_address_output);
  });
  size_t server_address_output_size;
  *serving_port = -1;

  TpuConfigurationApi_GetServerAddressAndPort_Params params;
  params.struct_size = TpuConfigurationApi_GetServerAddressAndPort_Params_SIZE;
  params.priv = nullptr;
  params.server_address_output_size = &server_address_output_size;
  params.server_address_output = &server_address_output;
  params.port_output = serving_port;
  params.status = status;

  tpu::OpsApiFn()->TpuConfigurationApi_GetServerAddressAndPortFn(&params);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status));
  *server_address =
      std::string(server_address_output, server_address_output_size);
  CHECK_NE(*serving_port, -1);
  return Status::OK();
}

TpuPodState::TpuPodState(
    int service_port, std::unique_ptr<TpuCompilationCacheService> cache_service)
    : cache_service_(std::move(cache_service)), service_port_(service_port) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "TpuPodState::TpuPodState");
}

TpuPodState::~TpuPodState() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_3(mht_3_v, 282, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "TpuPodState::~TpuPodState");

  if (cache_service_) {
    VLOG(1) << "Shutting down Compilation Cache Service.";
    if (cache_service_->Shutdown(20)) {
      if (service_port_ >= 0) {
        tpu::OpsApiFn()->TpuNetUtil_RecycleUnusedPortFn(service_port_);
      }
    } else {
      LOG(ERROR)
          << "Failed to shutdown Compilation Cache Service within timeout.";
    }
  }
  VLOG(1) << "Shutting down Compilation Cache Service done.";
}

string TpuPodState::DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "TpuPodState::DebugString");

  return "Wrapper for distributed TPU state";
}

Status GetTPUPodState(const ResourceMgr* rmgr, TpuPodState** pod_state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "GetTPUPodState");

  if (!rmgr) {
    return errors::Internal("No resource manager.");
  }
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    pod_state)
           .ok()) {
    return errors::FailedPrecondition(
        "The TPU system has not been initialized.");
  }
  return Status::OK();
}

bool HasTPUPodState(const ResourceMgr* rmgr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_6(mht_6_v, 323, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "HasTPUPodState");

  TpuPodState* pod_state;
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    &pod_state)
           .ok()) {
    return false;
  }
  pod_state->Unref();
  return true;
}

Status ConstructTpuPodState(
    ResourceMgr* rmgr, const std::vector<int32_t>& num_devices_per_host,
    tpu::TpuCompilationCacheInterface* compilation_cache,
    std::string* host_config_proto) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_pod_stateDTcc mht_7(mht_7_v, 340, "", "./tensorflow/core/tpu/kernels/tpu_pod_state.cc", "ConstructTpuPodState");

  TF_Status* status = TF_NewStatus();
  auto status_cleanup =
      absl::MakeCleanup([&status]() { TF_DeleteStatus(status); });

  int serving_port;
  std::string server_address;
  TF_RETURN_IF_ERROR(GetServerAddressAndPort(&server_address, &serving_port));

  char* host_config_output = nullptr;
  auto host_config_cleanup = absl::MakeCleanup([&host_config_output]() {
    tpu::OpsApiFn()->TpuConfigurationApi_FreeCharArrayFn(host_config_output);
  });
  size_t host_config_output_size;

  ConfigureDistributedTpuOp_DoWork_Params params;
  params.struct_size = ConfigureDistributedTpuOp_DoWork_Params_SIZE;
  params.priv = nullptr;
  params.num_cores_per_host_size = num_devices_per_host.size();
  params.num_cores_per_host = num_devices_per_host.data();
  params.server_address_size = server_address.size();
  params.server_address = server_address.data();
  params.host_config_output_size = &host_config_output_size;
  params.host_config_output = &host_config_output;
  params.status = status;

  tpu::OpsApiFn()->ConfigureDistributedTpuOp_DoWorkFn(&params);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status));
  *host_config_proto = std::string(host_config_output, host_config_output_size);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TpuCompilationCacheService> cache_service,
      ConstructCacheService(rmgr, serving_port, compilation_cache));

  // Delete TpuPodState if it exists, and recreate below.
  TF_RETURN_IF_ERROR(
      DeleteIfExists<TpuPodState>(rmgr, kTpuPodStateResourceName));
  return rmgr->Create(rmgr->default_container(), kTpuPodStateResourceName,
                      new TpuPodState(serving_port, std::move(cache_service)));
}
}  // namespace tensorflow
