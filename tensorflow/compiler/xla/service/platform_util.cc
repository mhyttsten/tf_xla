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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/platform_util.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Minimum supported CUDA compute capability is 3.5.
constexpr int kMinCudaComputeCapabilityMajor = 3;
constexpr int kMinCudaComputeCapabilityMinor = 5;

// The name of the interpreter platform.
constexpr char kInterpreter[] = "interpreter";

namespace {

std::string CanonicalPlatformName(const std::string& platform_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/service/platform_util.cc", "CanonicalPlatformName");

  std::string lowercase_platform_name = absl::AsciiStrToLower(platform_name);
  // "cpu" and "host" mean the same thing.
  if (lowercase_platform_name == "cpu") {
    return "host";
  }
  // When configured on CUDA, "gpu" and "cuda" mean the same thing.
  // When configured on ROCm, "gpu" and "rocm" mean the same thing.
  if (lowercase_platform_name == "gpu") {
#if TENSORFLOW_USE_ROCM
    return "rocm";
#else
    return "cuda";
#endif
  }
  return lowercase_platform_name;
}

StatusOr<std::vector<se::Platform*>> GetSupportedPlatforms() {
  return se::MultiPlatformManager::PlatformsWithFilter(
      [](const se::Platform* platform) {
        auto compiler_status = Compiler::GetForPlatform(platform);
        bool supported = compiler_status.ok();
        if (!supported) {
          LOG(INFO) << "platform " << platform->Name() << " present but no "
                    << "XLA compiler available: "
                    << compiler_status.status().error_message();
        }
        return supported;
      });
}

}  // namespace

/* static */ StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_1(mht_1_v, 253, "", "./tensorflow/compiler/xla/service/platform_util.cc", "PlatformUtil::GetSupportedPlatforms");

  // Gather all platforms which have an XLA compiler.
  return xla::GetSupportedPlatforms();
}

/* static */ StatusOr<se::Platform*> PlatformUtil::GetDefaultPlatform() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_2(mht_2_v, 261, "", "./tensorflow/compiler/xla/service/platform_util.cc", "PlatformUtil::GetDefaultPlatform");

  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());

  se::Platform* platform = nullptr;
  if (platforms.empty()) {
    return NotFound("no platforms found");
  } else if (platforms.size() == 1) {
    platform = platforms[0];
  } else if (platforms.size() == 2) {
    for (int i = 0; i < 2; i++) {
      if (absl::AsciiStrToLower(platforms[i]->Name()) == kInterpreter &&
          absl::AsciiStrToLower(platforms[1 - i]->Name()) != kInterpreter) {
        platform = platforms[1 - i];
        break;
      }
    }
  }
  if (platform != nullptr) {
    return platform;
  }

  // Multiple platforms present and we can't pick a reasonable default.
  std::string platforms_string = absl::StrJoin(
      platforms, ", ",
      [](std::string* out, const se::Platform* p) { out->append(p->Name()); });
  return InvalidArgument(
      "must specify platform because more than one platform (except for the "
      "interpreter platform) found: %s.",
      platforms_string);
}

/*static*/ StatusOr<se::Platform*> PlatformUtil::GetPlatform(
    const std::string& platform_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_3(mht_3_v, 297, "", "./tensorflow/compiler/xla/service/platform_util.cc", "PlatformUtil::GetPlatform");

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::MultiPlatformManager::PlatformWithName(
                          CanonicalPlatformName(platform_name)));
  TF_RETURN_IF_ERROR(Compiler::GetForPlatform(platform).status());
  return platform;
}

// Returns whether the device underlying the given StreamExecutor is supported
// by XLA.
static bool IsDeviceSupported(se::StreamExecutor* executor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_4(mht_4_v, 310, "", "./tensorflow/compiler/xla/service/platform_util.cc", "IsDeviceSupported");

  const auto& description = executor->GetDeviceDescription();
  if (executor->platform()->id() == se::cuda::kCudaPlatformId) {
    // CUDA devices must have a minimum compute capability.
    se::CudaComputeCapability cc = description.cuda_compute_capability();
    if (!cc.IsAtLeast(kMinCudaComputeCapabilityMajor,
                      kMinCudaComputeCapabilityMinor)) {
      LOG(INFO) << "StreamExecutor cuda device (" << executor->device_ordinal()
                << ") is of "
                << "insufficient compute capability: "
                << kMinCudaComputeCapabilityMajor << "."
                << kMinCudaComputeCapabilityMinor << " required, "
                << "device is " << cc.ToString();
      return false;
    }
  } else if (executor->platform()->id() == se::rocm::kROCmPlatformId) {
    auto rocm_compute_capability = description.rocm_compute_capability();
    if (!rocm_compute_capability.is_supported_gfx_version()) {
      LOG(INFO) << "StreamExecutor ROCM device (" << executor->device_ordinal()
                << ") is of unsupported "
                << "AMDGPU version : " << rocm_compute_capability.gfx_version()
                << ". The supported AMDGPU versions are "
                << rocm_compute_capability.supported_gfx_versions_str() << ".";
      return false;
    }
  }
  return true;
}

/* static */ StatusOr<std::vector<se::StreamExecutor*>>
PlatformUtil::GetStreamExecutors(
    se::Platform* platform,
    const absl::optional<std::set<int>>& allowed_devices) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSplatform_utilDTcc mht_5(mht_5_v, 345, "", "./tensorflow/compiler/xla/service/platform_util.cc", "PlatformUtil::GetStreamExecutors");

  int device_count = platform->VisibleDeviceCount();
  if (device_count <= 0) {
    return NotFound("no %s devices found", platform->Name());
  }
  if (platform->id() == se::host::kHostPlatformId) {
    // On host "devices", StreamExecutor exports a device for each hardware
    // thread. Because we parallelize a single computation across threads, it
    // doesn't make sense to expose these as separate devices, so by default we
    // fix the number of devices to one.  However we do let the user override
    // this behavior to help run tests on the host that run models in parallel
    // across multiple devices.
    device_count =
        GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
  }
  std::vector<se::StreamExecutor*> stream_executors(device_count, nullptr);
  VLOG(1) << "Initializing devices";
  {
    tensorflow::thread::ThreadPool thread_pool(
        tensorflow::Env::Default(), "device_initialization", device_count);
    for (int i = 0; i < device_count; ++i) {
      // Once a stream executor is instantiated it will cause allocations on
      // the device, for example for GPUs cuda context, cudnn handles etc. will
      // be constructed. By constructing stream executors only on the
      // allowed_devices, we don't make any allocations on other devices.
      // This helps in multi-process executions on the same host like horovod or
      // shared hosts.
      if (allowed_devices && allowed_devices->count(i) == 0) {
        VLOG(1) << "Not initializing StreamExecutor for device " << i
                << " since it is not in the visible device list";
        continue;
      }
      thread_pool.Schedule([platform, i, &stream_executors]() {
        VLOG(1) << "Started device init " << i;
        auto executor_status = platform->ExecutorForDevice(i);
        if (executor_status.ok()) {
          se::StreamExecutor* executor = executor_status.ValueOrDie();
          if (IsDeviceSupported(executor)) {
            stream_executors[i] = executor;
          }
        } else {
          LOG(WARNING) << "unable to create StreamExecutor for "
                       << platform->Name() << ":" << i << ": "
                       << executor_status.status().error_message();
        }
        VLOG(1) << "Finished device init " << i;
      });
    }
    // Block here in thread_pool destructor until all devices are initialized.
  }
  VLOG(1) << "Device initialization complete";

  std::vector<se::StreamExecutor*> out;
  for (se::StreamExecutor* executor : stream_executors) {
    if (executor != nullptr) {
      out.push_back(executor);
    }
  }
  if (out.empty()) {
    return InternalError("no supported devices found for platform %s",
                         platform->Name());
  }
  return out;
}

}  // namespace xla
