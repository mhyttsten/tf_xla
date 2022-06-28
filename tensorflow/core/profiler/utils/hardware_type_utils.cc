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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc() {
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

#include "tensorflow/core/profiler/utils/hardware_type_utils.h"

#include "absl/strings/match.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

// Get theoretical upperbound of single precision FMA throughput of the GPU per
// cycle per streaming multiprocessor.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
uint32 GetFmaMaxThroughputPerSMPerCycle(const DeviceCapabilities& device_cap) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/profiler/utils/hardware_type_utils.cc", "GetFmaMaxThroughputPerSMPerCycle");

  if (device_cap.device_vendor() == kDeviceVendorNvidia) {
    uint32 n_fp32_cores = 0;
    uint32 n_tc_cores = 0;
    switch (device_cap.compute_capability().major()) {
      case 2:
        // Fermi
        n_fp32_cores = 32;
        break;
      case 3:
        // Kepler
        n_fp32_cores = 192;
        break;
      case 5:
        // Maxwell
        n_fp32_cores = 128;
        break;
      case 6:
        // Pascal
        if (device_cap.compute_capability().minor() > 0) {
          // Pascal SM61/62
          n_fp32_cores = 128;
        } else {
          // Pascal SM60
          n_fp32_cores = 64;
        }
        break;
      case 7:
        // Volta and Turing
        n_fp32_cores = 64;
        n_tc_cores = 8;
        break;
      case 8:
        // Ampere
        if (device_cap.compute_capability().minor() >= 6) {
          // Ampere SM86
          n_fp32_cores = 128;
        } else {
          // Ampere SM80
          n_fp32_cores = 64;
        }
        n_tc_cores = 4;
        break;
      default:
        LOG(ERROR) << "Invalid GPU compute capability.";
        break;
    }
    // GPU TensorCore can execute 64 FMAs per cycle.
    // https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
    return n_fp32_cores + n_tc_cores * 64;
  } else if (device_cap.device_vendor() == kDeviceVendorAMD) {
    uint32_t n_xdlops = 0;
    uint32_t n_fp32_cores = 0;

    if (device_cap.compute_capability().major() <= 9) {
      n_fp32_cores = 64;
    } else {
      n_fp32_cores = 32;
    }
    // TODO(rocm-profiler): verify with new devices
    return n_fp32_cores + n_xdlops * 1;
  } else {
    LOG(ERROR) << "Unknown device vendor " << device_cap.device_vendor();
    return 0;
  }
}

}  // namespace

double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/profiler/utils/hardware_type_utils.cc", "GetFlopMaxThroughputPerSM");

  // One FMA = 2 floating point operations, one multiply and one add.
  return GetFmaMaxThroughputPerSMPerCycle(device_cap) * 2 *
         device_cap.clock_rate_in_ghz();
}

absl::string_view GpuModelName(const DeviceCapabilities& device_cap) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc mht_2(mht_2_v, 281, "", "./tensorflow/core/profiler/utils/hardware_type_utils.cc", "GpuModelName");

  if (device_cap.device_vendor() == kDeviceVendorNvidia) {
    switch (device_cap.compute_capability().major()) {
      case 2:
        return "Nvidia GPU (Fermi)";
      case 3:
        return "Nvidia GPU (Kepler)";
      case 5:
        return "Nvidia GPU (Maxwell)";
      case 6:
        return "Nvidia GPU (Pascal)";
      case 7:
        if (device_cap.compute_capability().minor() < 5) {
          return "Nvidia GPU (Volta)";
        } else {
          return "Nvidia GPU (Turing)";
        }
      case 8:
        return "Nvidia GPU (Ampere)";
      default:
        return "Nvidia GPU";
    }
  } else if (device_cap.device_vendor() == kDeviceVendorAMD) {
    switch (device_cap.compute_capability().major()) {
      case 9:
        return "AMD GPU - gfx-9XX series";
      case 10:
        return "AMD GPU - gfx-10XX series";
      case 11:
        return "AMD GPU - gfx-11XX series";
      default:
        return "AMD GPU";
    }
  } else {
    LOG(ERROR) << "Unknown device vendor " << device_cap.device_vendor();
    return "";
  }
}

HardwareType ParseHardwareType(absl::string_view device_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_type: \"" + std::string(device_type.data(), device_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc mht_3(mht_3_v, 324, "", "./tensorflow/core/profiler/utils/hardware_type_utils.cc", "ParseHardwareType");

  if (absl::StrContains(device_type, "GPU")) return HardwareType::GPU;
  if (device_type == "CPU") return HardwareType::CPU_ONLY;
  if (device_type == "TPU") return HardwareType::TPU;
  return HardwareType::UNKNOWN_HARDWARE;
}

bool HasDevice(HardwareType x) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPShardware_type_utilsDTcc mht_4(mht_4_v, 334, "", "./tensorflow/core/profiler/utils/hardware_type_utils.cc", "HasDevice");
 return x > tensorflow::profiler::CPU_ONLY; }

}  // namespace profiler
}  // namespace tensorflow
