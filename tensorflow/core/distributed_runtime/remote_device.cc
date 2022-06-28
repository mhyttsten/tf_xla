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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc() {
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

#include "tensorflow/core/distributed_runtime/remote_device.h"

#include <stdlib.h>

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class RemoteDevice : public Device {
 public:
  RemoteDevice(Env* env, const DeviceAttributes& da)
      : Device(env, da),
        local_dev_name_(DeviceNameUtils::LocalName(da.name())) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "RemoteDevice");
}

  Status Sync() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "Sync");
 return Status::OK(); }
  Allocator* GetAllocator(AllocatorAttributes attr) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "GetAllocator");
 return nullptr; }

  ResourceMgr* resource_manager() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "resource_manager");

    LOG(FATAL) << "Accessing the resource manager of a remote device is not "
               << "supported.";
    std::abort();
  }

  bool IsLocal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_4(mht_4_v, 232, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "IsLocal");
 return false; }

  bool IsRemoteCallAllowed() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_5(mht_5_v, 237, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "IsRemoteCallAllowed");
 return true; }

 private:
  const string local_dev_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteDevice);
};

void AsRemoteDevices(
    Env* env,
    const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
    LookupLocalDevice lookup_local_device,
    std::vector<std::unique_ptr<Device>>* remote_devices) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_6(mht_6_v, 252, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "AsRemoteDevices");

  for (const auto& da : device_attributes) {
    Device* local_device;
    if (lookup_local_device != nullptr &&
        lookup_local_device(da.name(), &local_device).ok()) {
      remote_devices->emplace_back(RenamedDevice::NewRenamedDevice(
          local_device->name(), local_device, false, false));
    } else {
      auto d = new RemoteDevice(env, da);
      remote_devices->emplace_back(d);
    }
  }
}

void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
                      const string& worker_name, NewRemoteDevicesDone done) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("worker_name: \"" + worker_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_7(mht_7_v, 271, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "NewRemoteDevices");

  WorkerInterface* wi = worker_cache->GetOrCreateWorker(worker_name);
  if (wi == nullptr) {
    std::vector<Device*> empty;
    done(errors::NotFound("Device ", worker_name, " is not found."), &empty);
    return;
  }
  struct Call {
    GetStatusRequest req;
    GetStatusResponse resp;
  };
  Call* call = new Call;
  auto cb = [env, worker_cache, worker_name, done, wi,
             call](const Status& status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_deviceDTcc mht_8(mht_8_v, 287, "", "./tensorflow/core/distributed_runtime/remote_device.cc", "lambda");

    Status s = status;
    std::vector<Device*> remote_devices;
    auto cleanup = gtl::MakeCleanup(
        [&worker_cache, &worker_name, &wi, &done, &remote_devices, &s, call] {
          worker_cache->ReleaseWorker(worker_name, wi);
          done(s, &remote_devices);
          delete call;
        });
    if (s.ok()) {
      DeviceNameUtils::ParsedName worker_name_parsed;
      if (!DeviceNameUtils::ParseFullName(worker_name, &worker_name_parsed) ||
          !worker_name_parsed.has_job || !worker_name_parsed.has_replica ||
          !worker_name_parsed.has_task) {
        s = errors::InvalidArgument("Could not parse worker name: ",
                                    worker_name);
        LOG(WARNING) << s;
        return;
      }
      remote_devices.reserve(call->resp.device_attributes_size());
      for (const DeviceAttributes& da : call->resp.device_attributes()) {
        DeviceNameUtils::ParsedName device_name_parsed;
        CHECK(DeviceNameUtils::ParseFullName(da.name(), &device_name_parsed))
            << "Device attribute name '" << da.name() << "' could not be "
            << "parsed. Device Attribute: " << da.DebugString();
        // Preserve the exact name, if possible.
        // TODO(b/37868888): Simplify when legacy device name formats removed.
        if (device_name_parsed.job == worker_name_parsed.job &&
            device_name_parsed.replica == worker_name_parsed.replica &&
            device_name_parsed.task == worker_name_parsed.task) {
          auto d = new RemoteDevice(env, da);
          remote_devices.push_back(d);
        } else {
          DeviceAttributes da_rewritten = da;
          da_rewritten.set_name(DeviceNameUtils::FullName(
              worker_name_parsed.job, worker_name_parsed.replica,
              worker_name_parsed.task, device_name_parsed.type,
              device_name_parsed.id));
          auto d = new RemoteDevice(env, da_rewritten);

          // Experimental: Skipping over adding any TPU-type devices that aren't
          // on the job called "worker" (but still adds the CPUs of other jobs).
          if (getenv("TPU_NO_POPULATE_DEVICE_LIST_FROM_CLUSTER_SPEC") !=
              nullptr) {
            if (worker_name_parsed.job == "worker" ||
                device_name_parsed.type.find("TPU") == std::string::npos) {
              remote_devices.push_back(d);
            }
          } else {
            remote_devices.push_back(d);
          }
        }
      }
    }
  };
  wi->GetStatusAsync(/*opts=*/nullptr, &call->req, &call->resp,
                     /*fail_fast=*/false, cb);
}

std::unique_ptr<Device> NewRemoteDevice(Env* env,
                                        DeviceAttributes device_attribute) {
  return std::make_unique<RemoteDevice>(env, device_attribute);
}

}  // namespace tensorflow
