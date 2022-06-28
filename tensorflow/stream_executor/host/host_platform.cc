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
class MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc() {
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

#include "tensorflow/stream_executor/host/host_platform.h"

#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/host/host_gpu_executor.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"

namespace stream_executor {
namespace host {

HostPlatform::HostPlatform() : name_("Host") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_0(mht_0_v, 201, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::HostPlatform");
}

HostPlatform::~HostPlatform() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_1(mht_1_v, 206, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::~HostPlatform");
}

Platform::Id HostPlatform::id() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_2(mht_2_v, 211, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::id");
 return kHostPlatformId; }

int HostPlatform::VisibleDeviceCount() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_3(mht_3_v, 216, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::VisibleDeviceCount");

  return std::thread::hardware_concurrency();
}

const std::string& HostPlatform::Name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_4(mht_4_v, 223, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::Name");
 return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
HostPlatform::DescriptionForDevice(int ordinal) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_5(mht_5_v, 229, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::DescriptionForDevice");

  return HostExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDevice(int ordinal) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_6(mht_6_v, 236, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::ExecutorForDevice");

  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_7(mht_7_v, 248, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::ExecutorForDeviceWithPluginConfig");

  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> HostPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_8(mht_8_v, 260, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::GetExecutor");

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
HostPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_9(mht_9_v, 269, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::GetUncachedExecutor");

  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<HostExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str()));
  }

  return std::move(executor);
}

void HostPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_10(mht_10_v, 289, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::RegisterTraceListener");

  LOG(FATAL) << "not yet implemented: register host trace listener";
}

void HostPlatform::UnregisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_11(mht_11_v, 296, "", "./tensorflow/stream_executor/host/host_platform.cc", "HostPlatform::UnregisterTraceListener");

  LOG(FATAL) << "not yet implemented: unregister host trace listener";
}

static void InitializeHostPlatform() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_platformDTcc mht_12(mht_12_v, 303, "", "./tensorflow/stream_executor/host/host_platform.cc", "InitializeHostPlatform");

  std::unique_ptr<Platform> platform(new host::HostPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace host
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(host_platform,
                            stream_executor::host::InitializeHostPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(host_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     host_platform);
