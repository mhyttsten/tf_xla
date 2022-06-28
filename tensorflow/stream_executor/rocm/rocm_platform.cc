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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/rocm/rocm_platform.h"

#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor {
namespace gpu {

ROCmPlatform::ROCmPlatform()
    : name_("ROCM"), min_numa_node_(0), limit_numa_node_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_0(mht_0_v, 201, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::ROCmPlatform");
}

ROCmPlatform::~ROCmPlatform() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_1(mht_1_v, 206, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::~ROCmPlatform");
}

// Due to legacy issues in user code, we can't currently call InpectNumaNodes
// at module initialization time, because non-GPU programs still include this
// plugin via various methods, so instead, it has to be init-on-reference.
void ROCmPlatform::InspectNumaNodes() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_2(mht_2_v, 214, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::InspectNumaNodes");

  // To get NUMA node information, we need to create all executors, so we can
  // examine their device descriptions to see their bus assignments.
  absl::once_flag once;
  absl::call_once(once, [&] {
    StreamExecutorConfig config;
    for (int i = 0; i < VisibleDeviceCount(); i++) {
      config.ordinal = i;
      StreamExecutor* exec = GetExecutor(config).ValueOrDie();
      if (i == 0) {
        // NUMA nodes may not start at 0, so set the minimum node  based on the
        // first executor we see.
        min_numa_node_ = exec->GetDeviceDescription().numa_node();
        limit_numa_node_ = min_numa_node_ + 1;
      } else {
        min_numa_node_ =
            std::min(min_numa_node_, exec->GetDeviceDescription().numa_node());
        limit_numa_node_ = std::max(
            limit_numa_node_, exec->GetDeviceDescription().numa_node() + 1);
      }
    }
  });
}

int ROCmPlatform::BusCount() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_3(mht_3_v, 241, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::BusCount");

  InspectNumaNodes();
  return limit_numa_node_ - min_numa_node_;
}

int ROCmPlatform::DeviceToBus(int device_ordinal) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_4(mht_4_v, 249, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::DeviceToBus");

  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  StreamExecutor* exec = GetExecutor(config).ValueOrDie();
  return exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

port::StatusOr<StreamExecutor*> ROCmPlatform::FirstExecutorForBus(
    int bus_ordinal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_5(mht_5_v, 260, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::FirstExecutorForBus");

  InspectNumaNodes();
  CHECK_LT(bus_ordinal, BusCount()) << "bus ordinal out of available range";
  for (int i = 0; i < VisibleDeviceCount(); i++) {
    if (DeviceToBus(i) == bus_ordinal) {
      StreamExecutorConfig config;
      config.ordinal = i;
      return GetExecutor(config).ValueOrDie();
    }
  }

  return port::Status{
      port::error::NOT_FOUND,
      absl::StrFormat("Executor for bus %d not found.", bus_ordinal)};
}

Platform::Id ROCmPlatform::id() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_6(mht_6_v, 279, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::id");
 return rocm::kROCmPlatformId; }

int ROCmPlatform::VisibleDeviceCount() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_7(mht_7_v, 284, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::VisibleDeviceCount");

  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.

  if (!gpu::GpuDriver::Init().ok()) {
    return -1;
  }

  return GpuDriver::GetDeviceCount();
}

const string& ROCmPlatform::Name() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_8(mht_8_v, 298, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::Name");
 return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
ROCmPlatform::DescriptionForDevice(int ordinal) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_9(mht_9_v, 304, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::DescriptionForDevice");

  return GpuExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDevice(int ordinal) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_10(mht_10_v, 311, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::ExecutorForDevice");

  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_11(mht_11_v, 323, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::ExecutorForDeviceWithPluginConfig");

  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ROCmPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_12(mht_12_v, 335, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::GetExecutor");

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
ROCmPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_13(mht_13_v, 344, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::GetUncachedExecutor");

  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<GpuExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for ROCM device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }

  return std::move(executor);
}

void ROCmPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_14(mht_14_v, 364, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::RegisterTraceListener");

  LOG(FATAL) << "not yet implemented: register ROCM trace listener";
}

void ROCmPlatform::UnregisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_15(mht_15_v, 371, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "ROCmPlatform::UnregisterTraceListener");

  LOG(FATAL) << "not yet implemented: unregister ROCM trace listener";
}

}  // namespace gpu

static void InitializeROCmPlatform() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_platformDTcc mht_16(mht_16_v, 380, "", "./tensorflow/stream_executor/rocm/rocm_platform.cc", "InitializeROCmPlatform");

  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  auto status = MultiPlatformManager::PlatformWithName("ROCM");
  if (!status.ok()) {
    std::unique_ptr<gpu::ROCmPlatform> platform(new gpu::ROCmPlatform);
    SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(rocm_platform,
                            stream_executor::InitializeROCmPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(rocm_platform, multi_platform_manager);
