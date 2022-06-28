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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"

#include <algorithm>
#include <map>
#include <string>

#include "absl/strings/ascii.h"

namespace tflite {
namespace gpu {
namespace {

GpuVendor GetGpuVendor(const std::string& gpu_description) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GetGpuVendor");

  const std::map<std::string, GpuVendor> kMapping = {
      {"adreno", GpuVendor::kQualcomm},
      {"apple", GpuVendor::kApple},
      {"qualcomm", GpuVendor::kQualcomm},
      {"mali", GpuVendor::kMali},
      {"powervr", GpuVendor::kPowerVR},
      {"advanced micro devices", GpuVendor::kAMD},
      {"intel", GpuVendor::kIntel},
      {"nvidia", GpuVendor::kNvidia},
      {"amd", GpuVendor::kAMD},
      {"radeon", GpuVendor::kAMD},
      {"power", GpuVendor::kPowerVR},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return GpuVendor::kUnknown;
}

AdrenoGpu GetAdrenoGpuVersion(const std::string& gpu_description) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GetAdrenoGpuVersion");

  const std::map<std::string, AdrenoGpu> kMapping = {
      // Adreno 7xx series
      {"730", AdrenoGpu::kAdreno730},
      // Adreno 6xx series
      {"685", AdrenoGpu::kAdreno685},
      {"680", AdrenoGpu::kAdreno680},
      {"675", AdrenoGpu::kAdreno675},
      {"660", AdrenoGpu::kAdreno660},
      {"650", AdrenoGpu::kAdreno650},
      {"640", AdrenoGpu::kAdreno640},
      {"630", AdrenoGpu::kAdreno630},
      {"620", AdrenoGpu::kAdreno620},
      {"618", AdrenoGpu::kAdreno618},
      {"616", AdrenoGpu::kAdreno616},
      {"615", AdrenoGpu::kAdreno615},
      {"612", AdrenoGpu::kAdreno612},
      {"610", AdrenoGpu::kAdreno610},
      {"605", AdrenoGpu::kAdreno605},
      // Adreno 5xx series
      {"540", AdrenoGpu::kAdreno540},
      {"530", AdrenoGpu::kAdreno530},
      {"512", AdrenoGpu::kAdreno512},
      {"510", AdrenoGpu::kAdreno510},
      {"509", AdrenoGpu::kAdreno509},
      {"508", AdrenoGpu::kAdreno508},
      {"506", AdrenoGpu::kAdreno506},
      {"505", AdrenoGpu::kAdreno505},
      {"504", AdrenoGpu::kAdreno504},
      // Adreno 4xx series
      {"430", AdrenoGpu::kAdreno430},
      {"420", AdrenoGpu::kAdreno420},
      {"418", AdrenoGpu::kAdreno418},
      {"405", AdrenoGpu::kAdreno405},
      // Adreno 3xx series
      {"330", AdrenoGpu::kAdreno330},
      {"320", AdrenoGpu::kAdreno320},
      {"308", AdrenoGpu::kAdreno308},
      {"306", AdrenoGpu::kAdreno306},
      {"305", AdrenoGpu::kAdreno305},
      {"304", AdrenoGpu::kAdreno304},
      // Adreno 2xx series
      {"225", AdrenoGpu::kAdreno225},
      {"220", AdrenoGpu::kAdreno220},
      {"205", AdrenoGpu::kAdreno205},
      {"203", AdrenoGpu::kAdreno203},
      {"200", AdrenoGpu::kAdreno200},
      // Adreno 1xx series
      {"130", AdrenoGpu::kAdreno130},
      {"120", AdrenoGpu::kAdreno120},
  };

  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return AdrenoGpu::kUnknown;
}

MaliGpu GetMaliGpuVersion(const std::string& gpu_description) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_2(mht_2_v, 288, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GetMaliGpuVersion");

  // Order must be preserved
  const std::vector<std::pair<std::string, MaliGpu>> kMapping = {
      {"t604", MaliGpu::kT604}, {"t622", MaliGpu::kT622},
      {"t624", MaliGpu::kT624}, {"t628", MaliGpu::kT628},
      {"t658", MaliGpu::kT658}, {"t678", MaliGpu::kT678},
      {"t720", MaliGpu::kT720}, {"t760", MaliGpu::kT760},
      {"t820", MaliGpu::kT820}, {"t830", MaliGpu::kT830},
      {"t860", MaliGpu::kT860}, {"t880", MaliGpu::kT880},
      {"g310", MaliGpu::kG310}, {"g31", MaliGpu::kG31},
      {"g510", MaliGpu::kG510}, {"g51", MaliGpu::kG51},
      {"g52", MaliGpu::kG52},   {"g57", MaliGpu::kG57},
      {"g610", MaliGpu::kG610}, {"g68", MaliGpu::kG68},
      {"g710", MaliGpu::kG710}, {"g71", MaliGpu::kG71},
      {"g72", MaliGpu::kG72},   {"g76", MaliGpu::kG76},
      {"g77", MaliGpu::kG77},   {"g78", MaliGpu::kG78},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return MaliGpu::kUnknown;
}

}  // namespace

AdrenoInfo::AdrenoInfo(const std::string& device_version)
    : adreno_gpu(GetAdrenoGpuVersion(device_version)) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_version: \"" + device_version + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_3(mht_3_v, 320, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::AdrenoInfo");
}

bool AdrenoInfo::IsAdreno1xx() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_4(mht_4_v, 325, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno1xx");

  return adreno_gpu == AdrenoGpu::kAdreno120 ||
         adreno_gpu == AdrenoGpu::kAdreno130;
}

bool AdrenoInfo::IsAdreno2xx() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_5(mht_5_v, 333, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno2xx");

  return adreno_gpu == AdrenoGpu::kAdreno200 ||
         adreno_gpu == AdrenoGpu::kAdreno203 ||
         adreno_gpu == AdrenoGpu::kAdreno205 ||
         adreno_gpu == AdrenoGpu::kAdreno220 ||
         adreno_gpu == AdrenoGpu::kAdreno225;
}

bool AdrenoInfo::IsAdreno3xx() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_6(mht_6_v, 344, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno3xx");

  return adreno_gpu == AdrenoGpu::kAdreno304 ||
         adreno_gpu == AdrenoGpu::kAdreno305 ||
         adreno_gpu == AdrenoGpu::kAdreno306 ||
         adreno_gpu == AdrenoGpu::kAdreno308 ||
         adreno_gpu == AdrenoGpu::kAdreno320 ||
         adreno_gpu == AdrenoGpu::kAdreno330;
}

bool AdrenoInfo::IsAdreno4xx() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_7(mht_7_v, 356, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno4xx");

  return adreno_gpu == AdrenoGpu::kAdreno405 ||
         adreno_gpu == AdrenoGpu::kAdreno418 ||
         adreno_gpu == AdrenoGpu::kAdreno420 ||
         adreno_gpu == AdrenoGpu::kAdreno430;
}

bool AdrenoInfo::IsAdreno5xx() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_8(mht_8_v, 366, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno5xx");

  return adreno_gpu == AdrenoGpu::kAdreno504 ||
         adreno_gpu == AdrenoGpu::kAdreno505 ||
         adreno_gpu == AdrenoGpu::kAdreno506 ||
         adreno_gpu == AdrenoGpu::kAdreno508 ||
         adreno_gpu == AdrenoGpu::kAdreno509 ||
         adreno_gpu == AdrenoGpu::kAdreno510 ||
         adreno_gpu == AdrenoGpu::kAdreno512 ||
         adreno_gpu == AdrenoGpu::kAdreno530 ||
         adreno_gpu == AdrenoGpu::kAdreno540;
}

bool AdrenoInfo::IsAdreno6xx() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_9(mht_9_v, 381, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno6xx");

  return adreno_gpu == AdrenoGpu::kAdreno605 ||
         adreno_gpu == AdrenoGpu::kAdreno610 ||
         adreno_gpu == AdrenoGpu::kAdreno612 ||
         adreno_gpu == AdrenoGpu::kAdreno615 ||
         adreno_gpu == AdrenoGpu::kAdreno616 ||
         adreno_gpu == AdrenoGpu::kAdreno618 ||
         adreno_gpu == AdrenoGpu::kAdreno620 ||
         adreno_gpu == AdrenoGpu::kAdreno630 ||
         adreno_gpu == AdrenoGpu::kAdreno640 ||
         adreno_gpu == AdrenoGpu::kAdreno650 ||
         adreno_gpu == AdrenoGpu::kAdreno660 ||
         adreno_gpu == AdrenoGpu::kAdreno675 ||
         adreno_gpu == AdrenoGpu::kAdreno680 ||
         adreno_gpu == AdrenoGpu::kAdreno685;
}

bool AdrenoInfo::IsAdreno7xx() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_10(mht_10_v, 401, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno7xx");

  return adreno_gpu == AdrenoGpu::kAdreno730;
}

bool AdrenoInfo::IsAdreno6xxOrHigher() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_11(mht_11_v, 408, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::IsAdreno6xxOrHigher");

  return (!compiler_bugs_in_a6xx && IsAdreno6xx()) || IsAdreno7xx();
}

int AdrenoInfo::GetMaximumWavesCount() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_12(mht_12_v, 415, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::GetMaximumWavesCount");

  if (IsAdreno7xx()) {
    return 16;
  } else if (IsAdreno6xx()) {
    if (adreno_gpu == AdrenoGpu::kAdreno640) {
      return 30;
    } else {
      return 16;
    }
  } else {
    // all other versions not supported
    return 1;
  }
}

int AdrenoInfo::GetRegisterMemorySizePerComputeUnit() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_13(mht_13_v, 433, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::GetRegisterMemorySizePerComputeUnit");

  if (IsAdreno7xx()) {
    return 128 * 96 * 16;
  } else if (IsAdreno6xx()) {
    if (adreno_gpu == AdrenoGpu::kAdreno640) {
      return 128 * 144 * 16;
    } else if (adreno_gpu == AdrenoGpu::kAdreno620 ||
               adreno_gpu == AdrenoGpu::kAdreno650 ||
               adreno_gpu == AdrenoGpu::kAdreno660) {
      return 128 * 64 * 16;
    } else {
      return 128 * 96 * 16;
    }
  } else {
    // all other versions not supported
    return 1;
  }
}

int AdrenoInfo::GetMaximumWavesCount(int register_footprint_per_tread,
                                     bool full_wave) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_14(mht_14_v, 456, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::GetMaximumWavesCount");

  const int register_usage_per_wave =
      GetWaveSize(full_wave) * register_footprint_per_tread;
  const int possible_waves_count =
      GetRegisterMemorySizePerComputeUnit() / register_usage_per_wave;
  return std::min(possible_waves_count, GetMaximumWavesCount());
}

int AdrenoInfo::GetWaveSize(bool full_wave) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_15(mht_15_v, 467, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::GetWaveSize");

  if (IsAdreno7xx()) {
    return full_wave ? 128 : 64;
  } else if (IsAdreno6xx()) {
    return full_wave ? 128 : 64;
  } else if (IsAdreno5xx() || IsAdreno4xx()) {
    return full_wave ? 64 : 32;
  } else {
    return full_wave ? 32 : 16;
  }
}

int AdrenoInfo::GetComputeUnitsCount() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_16(mht_16_v, 482, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AdrenoInfo::GetComputeUnitsCount");

  // can provide not correct numbers.
  switch (adreno_gpu) {
    // Adreno 7xx series
    case AdrenoGpu::kAdreno730:
      return 4;
    // Adreno 6xx series
    case AdrenoGpu::kAdreno685:
      return 4;
    case AdrenoGpu::kAdreno680:
      return 4;
    case AdrenoGpu::kAdreno675:
      return 4;
    case AdrenoGpu::kAdreno660:
      return 3;
    case AdrenoGpu::kAdreno650:
      return 3;
    case AdrenoGpu::kAdreno640:
      return 2;
    case AdrenoGpu::kAdreno630:
      return 2;
    case AdrenoGpu::kAdreno620:
      return 1;
    case AdrenoGpu::kAdreno618:
      return 1;
    case AdrenoGpu::kAdreno616:
      return 1;
    case AdrenoGpu::kAdreno615:
      return 1;
    case AdrenoGpu::kAdreno612:
      return 1;
    case AdrenoGpu::kAdreno610:
      return 1;
    case AdrenoGpu::kAdreno605:
      return 1;
    // Adreno 5xx series
    case AdrenoGpu::kAdreno540:
      return 4;
    case AdrenoGpu::kAdreno530:
      return 4;
    case AdrenoGpu::kAdreno512:
      return 2;
    case AdrenoGpu::kAdreno510:
      return 2;
    case AdrenoGpu::kAdreno509:
      return 2;
    case AdrenoGpu::kAdreno508:
      return 1;
    case AdrenoGpu::kAdreno506:
      return 1;
    case AdrenoGpu::kAdreno505:
      return 1;
    case AdrenoGpu::kAdreno504:
      return 1;
    // Adreno 4xx series
    case AdrenoGpu::kAdreno430:
      return 4;
    case AdrenoGpu::kAdreno420:
      return 4;
    case AdrenoGpu::kAdreno418:
      return 2;
    case AdrenoGpu::kAdreno405:
      return 1;
    // Adreno 3xx series
    case AdrenoGpu::kAdreno330:
      return 4;
    case AdrenoGpu::kAdreno320:
      return 2;
    case AdrenoGpu::kAdreno308:
      return 1;
    case AdrenoGpu::kAdreno306:
      return 1;
    case AdrenoGpu::kAdreno305:
      return 1;
    case AdrenoGpu::kAdreno304:
      return 1;
    default:
      return 1;
  }
}

AppleInfo::AppleInfo(const std::string& gpu_description) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_17(mht_17_v, 567, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::AppleInfo");

  const std::map<std::string, AppleGpu> kMapping = {
      {"apple a7 gpu", AppleGpu::kA7},
      {"apple a8 gpu", AppleGpu::kA8},
      {"apple a8x gpu", AppleGpu::kA8X},
      {"apple a9 gpu", AppleGpu::kA9},
      {"apple a9x gpu", AppleGpu::kA9X},
      {"apple a10 gpu", AppleGpu::kA10},
      {"apple a10x gpu", AppleGpu::kA10X},
      {"apple a11 gpu", AppleGpu::kA11},
      {"apple a12 gpu", AppleGpu::kA12},
      {"apple a12x gpu", AppleGpu::kA12X},
      {"apple a12z gpu", AppleGpu::kA12Z},
      {"apple a13 gpu", AppleGpu::kA13},
      {"apple a14 gpu", AppleGpu::kA14},
      {"apple a15 gpu", AppleGpu::kA15},
      // on tablets we have metal device name "apple m1 gpu"
      // and on notebooks "apple m1"
      {"apple m1 gpu", AppleGpu::kM1},
      {"apple m1", AppleGpu::kM1},
      {"apple m1 pro", AppleGpu::kM1Pro},
      {"apple m1 max", AppleGpu::kM1Max},
  };
  auto it = kMapping.find(gpu_description);
  if (it != kMapping.end()) {
    gpu_type = it->second;
  } else {
    gpu_type = AppleGpu::kUnknown;
  }
}

bool AppleInfo::IsA7GenerationGpu() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_18(mht_18_v, 601, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsA7GenerationGpu");
 return gpu_type == AppleGpu::kA7; }
bool AppleInfo::IsA8GenerationGpu() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_19(mht_19_v, 605, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsA8GenerationGpu");

  return gpu_type == AppleGpu::kA8 || gpu_type == AppleGpu::kA8X;
}

bool AppleInfo::IsLocalMemoryPreferredOverGlobal() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_20(mht_20_v, 612, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsLocalMemoryPreferredOverGlobal");

  return IsA7GenerationGpu() || IsA8GenerationGpu();
}

bool AppleInfo::IsBionic() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_21(mht_21_v, 619, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsBionic");

  return gpu_type == AppleGpu::kA11 || gpu_type == AppleGpu::kA12 ||
         gpu_type == AppleGpu::kA12X || gpu_type == AppleGpu::kA12Z ||
         gpu_type == AppleGpu::kA13 || gpu_type == AppleGpu::kA14 ||
         gpu_type == AppleGpu::kA15 || gpu_type == AppleGpu::kM1 ||
         gpu_type == AppleGpu::kM1Pro || gpu_type == AppleGpu::kM1Max;
}

bool AppleInfo::IsSIMDMatMulSupported() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_22(mht_22_v, 630, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsSIMDMatMulSupported");

  return gpu_type == AppleGpu::kA14 || gpu_type == AppleGpu::kA15 ||
         gpu_type == AppleGpu::kM1 || gpu_type == AppleGpu::kM1Pro ||
         gpu_type == AppleGpu::kM1Max;
}

bool AppleInfo::IsSIMDMatMulFp32Perf2x() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_23(mht_23_v, 639, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsSIMDMatMulFp32Perf2x");

  return gpu_type == AppleGpu::kA15;
}

bool AppleInfo::IsRoundToNearestSupported() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_24(mht_24_v, 646, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::IsRoundToNearestSupported");
 return IsBionic(); }

int AppleInfo::GetComputeUnitsCount() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_25(mht_25_v, 651, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::GetComputeUnitsCount");

  switch (gpu_type) {
    case AppleGpu::kA7:
      return 4;
    case AppleGpu::kA8:
      return 4;
    case AppleGpu::kA8X:
      return 8;
    case AppleGpu::kA9:
      return 6;
    case AppleGpu::kA9X:
      return 12;
    case AppleGpu::kA10:
      return 6;
    case AppleGpu::kA10X:
      return 12;
    case AppleGpu::kA11:
      return 3;
    case AppleGpu::kA12:
      return 4;
    case AppleGpu::kA12X:
      return 7;
    case AppleGpu::kA12Z:
      return 8;
    case AppleGpu::kA13:
      return 4;
    case AppleGpu::kA14:
      return 4;
    // For A15, M1, M1 Pro and M1 Max we can not receive exact CU count from
    // name. No official Metal API to receive this info.
    case AppleGpu::kA15:
      if (compute_units != -1) {
        return compute_units;
      }
      return 5;
    case AppleGpu::kM1:
      // approximate, can be 7 or 8
      return 8;
    case AppleGpu::kM1Pro:
      // approximate, can be 14 or 16
      return 16;
    case AppleGpu::kM1Max:
      // approximate, can be 24 or 32
      return 32;
    case AppleGpu::kUnknown:
      return 4;
  }
}

void AppleInfo::SetComputeUnits(int compute_units_count) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_26(mht_26_v, 703, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "AppleInfo::SetComputeUnits");

  compute_units = compute_units_count;
}

MaliInfo::MaliInfo(const std::string& gpu_description)
    : gpu_version(GetMaliGpuVersion(gpu_description)) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_27(mht_27_v, 712, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::MaliInfo");
}

bool MaliInfo::IsMaliT6xx() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_28(mht_28_v, 717, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsMaliT6xx");

  return gpu_version == MaliGpu::kT604 || gpu_version == MaliGpu::kT622 ||
         gpu_version == MaliGpu::kT624 || gpu_version == MaliGpu::kT628 ||
         gpu_version == MaliGpu::kT658 || gpu_version == MaliGpu::kT678;
}

bool MaliInfo::IsMaliT7xx() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_29(mht_29_v, 726, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsMaliT7xx");

  return gpu_version == MaliGpu::kT720 || gpu_version == MaliGpu::kT760;
}

bool MaliInfo::IsMaliT8xx() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_30(mht_30_v, 733, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsMaliT8xx");

  return gpu_version == MaliGpu::kT820 || gpu_version == MaliGpu::kT830 ||
         gpu_version == MaliGpu::kT860 || gpu_version == MaliGpu::kT880;
}

bool MaliInfo::IsMidgard() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_31(mht_31_v, 741, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsMidgard");

  return IsMaliT6xx() || IsMaliT7xx() || IsMaliT8xx();
}

bool MaliInfo::IsBifrostGen1() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_32(mht_32_v, 748, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsBifrostGen1");

  return gpu_version == MaliGpu::kG31 || gpu_version == MaliGpu::kG51 ||
         gpu_version == MaliGpu::kG71;
}

bool MaliInfo::IsBifrostGen2() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_33(mht_33_v, 756, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsBifrostGen2");

  return gpu_version == MaliGpu::kG52 || gpu_version == MaliGpu::kG72;
}

bool MaliInfo::IsBifrostGen3() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_34(mht_34_v, 763, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsBifrostGen3");
 return gpu_version == MaliGpu::kG76; }

bool MaliInfo::IsBifrost() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_35(mht_35_v, 768, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsBifrost");

  return IsBifrostGen1() || IsBifrostGen2() || IsBifrostGen3();
}

bool MaliInfo::IsValhallGen1() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_36(mht_36_v, 775, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsValhallGen1");

  return gpu_version == MaliGpu::kG57 || gpu_version == MaliGpu::kG77;
}

bool MaliInfo::IsValhallGen2() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_37(mht_37_v, 782, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsValhallGen2");

  return gpu_version == MaliGpu::kG68 || gpu_version == MaliGpu::kG78;
}

bool MaliInfo::IsValhallGen3() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_38(mht_38_v, 789, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsValhallGen3");

  return gpu_version == MaliGpu::kG310 || gpu_version == MaliGpu::kG510 ||
         gpu_version == MaliGpu::kG610 || gpu_version == MaliGpu::kG710;
}

bool MaliInfo::IsValhall() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_39(mht_39_v, 797, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "MaliInfo::IsValhall");

  return IsValhallGen1() || IsValhallGen2() || IsValhallGen3();
}

void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("gpu_description: \"" + gpu_description + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_40(mht_40_v, 806, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GetGpuInfoFromDeviceDescription");

  gpu_info->gpu_api = gpu_api;
  std::string lowered = gpu_description;
  absl::AsciiStrToLower(&lowered);
  gpu_info->vendor = GetGpuVendor(lowered);
  if (gpu_info->IsAdreno()) {
    gpu_info->adreno_info = AdrenoInfo(lowered);
  } else if (gpu_info->IsApple()) {
    gpu_info->apple_info = AppleInfo(lowered);
    gpu_info->supported_subgroup_sizes = {32};
  } else if (gpu_info->IsMali()) {
    gpu_info->mali_info = MaliInfo(lowered);
  }
}

std::string OpenClVersionToString(OpenClVersion version) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_41(mht_41_v, 824, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "OpenClVersionToString");

  switch (version) {
    case OpenClVersion::kCl1_0:
      return "1.0";
    case OpenClVersion::kCl1_1:
      return "1.1";
    case OpenClVersion::kCl1_2:
      return "1.2";
    case OpenClVersion::kCl2_0:
      return "2.0";
    case OpenClVersion::kCl2_1:
      return "2.1";
    case OpenClVersion::kCl2_2:
      return "2.2";
    case OpenClVersion::kCl3_0:
      return "3.0";
    default:
      return "Unknown OpenCL version";
  }
}

bool OpenGlInfo::SupportsExplicitFp16() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_42(mht_42_v, 848, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "OpenGlInfo::SupportsExplicitFp16");

  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "GL_EXT_shader_explicit_arithmetic_types_float16") {
      supports_f16_alu = true;
    }
    if (ext == "GL_EXT_shader_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool VulkanInfo::SupportsExplicitFp16() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_43(mht_43_v, 865, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "VulkanInfo::SupportsExplicitFp16");

  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "VK_KHR_shader_float16_int8") {
      supports_f16_alu = true;
    }
    if (ext == "VK_KHR_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool OpenClInfo::IsImage2dFromBufferSupported() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_44(mht_44_v, 882, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "OpenClInfo::IsImage2dFromBufferSupported");

  if (image_pitch_alignment == 0) {
    return false;
  }
  if (image_base_address_alignment == 0) {
    return false;
  }
  if (cl_version == OpenClVersion::kCl2_0 ||
      cl_version == OpenClVersion::kCl2_1 ||
      cl_version == OpenClVersion::kCl2_2) {
    return true;
  }
  for (const auto& ext : extensions) {
    if (ext == "cl_khr_image2d_from_buffer") {
      return true;
    }
  }
  return false;
}

bool GpuInfo::IsAdreno() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_45(mht_45_v, 905, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsAdreno");
 return vendor == GpuVendor::kQualcomm; }

bool GpuInfo::IsApple() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_46(mht_46_v, 910, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApple");
 return vendor == GpuVendor::kApple; }

bool GpuInfo::IsMali() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_47(mht_47_v, 915, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsMali");
 return vendor == GpuVendor::kMali; }

bool GpuInfo::IsPowerVR() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_48(mht_48_v, 920, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsPowerVR");
 return vendor == GpuVendor::kPowerVR; }

bool GpuInfo::IsNvidia() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_49(mht_49_v, 925, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsNvidia");
 return vendor == GpuVendor::kNvidia; }

bool GpuInfo::IsAMD() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_50(mht_50_v, 930, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsAMD");
 return vendor == GpuVendor::kAMD; }

bool GpuInfo::IsIntel() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_51(mht_51_v, 935, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsIntel");
 return vendor == GpuVendor::kIntel; }

bool GpuInfo::IsRoundToNearestSupported() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_52(mht_52_v, 940, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsRoundToNearestSupported");

  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16_rtn || opencl_info.supports_fp32_rtn;
  }
  if (IsApple()) {
    return apple_info.IsRoundToNearestSupported();
  }
  if (IsAdreno()) {
    if (adreno_info.IsAdreno1xx() || adreno_info.IsAdreno2xx() ||
        adreno_info.IsAdreno3xx()) {
      return false;
    }
  }
  if (IsPowerVR()) {
    return false;
  }
  return true;
}

bool GpuInfo::SupportsFP16() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_53(mht_53_v, 962, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsFP16");

  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16;
  }
  return true;
}

bool GpuInfo::SupportsTextureArray() const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_54(mht_54_v, 972, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsTextureArray");

  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  return true;
}

bool GpuInfo::SupportsImageBuffer() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_55(mht_55_v, 985, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsImageBuffer");

  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  return true;
}

bool GpuInfo::SupportsImage3D() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_56(mht_56_v, 998, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsImage3D");

  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    if (IsMali() && mali_info.IsMidgard()) {
      // On Mali T880 read_imageh doesn't compile with image3d_t
      return false;
    }
    return opencl_info.supports_image3d_writes;
  }
  return true;
}

bool GpuInfo::SupportsImages() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_57(mht_57_v, 1015, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsImages");

  if (IsApiOpenCl()) {
    return opencl_info.supports_images;
  }
  return true;
}

bool GpuInfo::SupportsPointersInKernels() const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_58(mht_58_v, 1025, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsPointersInKernels");

  return IsApiOpenCl() || IsApiMetal();
}

bool GpuInfo::IsWaveSizeEqualTo32() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_59(mht_59_v, 1032, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsWaveSizeEqualTo32");

  return supported_subgroup_sizes.size() == 1 &&
         supported_subgroup_sizes[0] == 32;
}

bool GpuInfo::SupportsExtension(const std::string& extension) const {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("extension: \"" + extension + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_60(mht_60_v, 1041, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsExtension");

  const std::vector<std::string>* extensions = nullptr;
  if (IsApiOpenGl()) {
    extensions = &opengl_info.extensions;
  } else if (IsApiVulkan()) {
    extensions = &vulkan_info.extensions;
  } else if (IsApiOpenCl()) {
    extensions = &opencl_info.extensions;
  }
  if (!extensions) {
    return false;
  }
  for (const auto& ext : *extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsSubGroupWithSize(int sub_group_size) const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_61(mht_61_v, 1064, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsSubGroupWithSize");

  for (auto subgroup_size : supported_subgroup_sizes) {
    if (sub_group_size == subgroup_size) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsFloatImage2D(DataType data_type, int channels) const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_62(mht_62_v, 1076, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::SupportsFloatImage2D");

  if (IsApiOpenCl()) {
    if (channels == 1) {
      return data_type == DataType::FLOAT32 ? opencl_info.supports_r_f32_tex2d
                                            : opencl_info.supports_r_f16_tex2d;
    } else if (channels == 2) {
      return data_type == DataType::FLOAT32 ? opencl_info.supports_rg_f32_tex2d
                                            : opencl_info.supports_rg_f16_tex2d;
    } else if (channels == 3) {
      return data_type == DataType::FLOAT32
                 ? opencl_info.supports_rgb_f32_tex2d
                 : opencl_info.supports_rgb_f16_tex2d;
    } else if (channels == 4) {
      return data_type == DataType::FLOAT32
                 ? opencl_info.supports_rgba_f32_tex2d
                 : opencl_info.supports_rgba_f16_tex2d;
    } else {
      return false;
    }
  }
  return false;
}

int GpuInfo::GetComputeUnitsCount() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_63(mht_63_v, 1102, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetComputeUnitsCount");

  if (IsApiOpenCl()) {
    return opencl_info.compute_units_count;
  }
  if (IsApple()) {
    return apple_info.GetComputeUnitsCount();
  }
  if (IsAMD() && IsApiVulkan()) {
    return amd_info.GetComputeUnitsCount();
  }
  if (IsAdreno()) {
    return adreno_info.GetComputeUnitsCount();
  }
  return 1;
}

int GpuInfo::GetMaxWorkGroupSizeForX() const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_64(mht_64_v, 1121, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxWorkGroupSizeForX");

  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_x;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_x;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_x;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_x;
  }
  return 256;
}

int GpuInfo::GetMaxWorkGroupSizeForY() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_65(mht_65_v, 1140, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxWorkGroupSizeForY");

  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_y;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_y;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_y;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_y;
  }
  return 256;
}

int GpuInfo::GetMaxWorkGroupSizeForZ() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_66(mht_66_v, 1159, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxWorkGroupSizeForZ");

  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_z;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_z;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_z;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_z;
  }
  return 64;
}

int GpuInfo::GetMaxWorkGroupTotalSize() const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_67(mht_67_v, 1178, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxWorkGroupTotalSize");

  if (IsApiOpenGl()) {
    return opengl_info.max_work_group_invocations;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_invocations;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_total_size;
  }
  if (IsApiMetal()) {
    int max_size = metal_info.max_work_group_size_x;
    max_size = std::max(max_size, metal_info.max_work_group_size_y);
    max_size = std::max(max_size, metal_info.max_work_group_size_z);
    return max_size;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage2DWidth() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_68(mht_68_v, 1200, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage2DWidth");

  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_width;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_width;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DHeight() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_69(mht_69_v, 1219, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage2DHeight");

  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_height;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_height;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DArrayLayers() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_70(mht_70_v, 1238, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage2DArrayLayers");

  if (IsApiOpenGl()) {
    return opengl_info.max_array_texture_layers;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_array_layers;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image_array_max_layers;
  }
  if (IsApiMetal()) {
    return metal_info.image_array_max_layers;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DWidth() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_71(mht_71_v, 1257, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage3DWidth");

  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_width;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_width;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DHeight() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_72(mht_72_v, 1271, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage3DHeight");

  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_height;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_height;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DDepth() const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_73(mht_73_v, 1285, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImage3DDepth");

  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_depth;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_depth;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxBufferSize() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_74(mht_74_v, 1299, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxBufferSize");

  if (IsApiOpenCl()) {
    return opencl_info.buffer_max_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxMemoryAllocationSize() const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_75(mht_75_v, 1313, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxMemoryAllocationSize");

  if (IsApiOpenCl()) {
    return opencl_info.max_allocation_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxImageBufferWidth() const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_76(mht_76_v, 1327, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImageBufferWidth");

  if (IsApiOpenCl()) {
    return opencl_info.image_buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_texel_buffer_elements;
  }
  return 64 * 1024;
}

int GpuInfo::GetMaxImageArguments() const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_77(mht_77_v, 1339, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::GetMaxImageArguments");

  if (IsApiOpenGl()) {
    return opengl_info.max_image_units;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_sampled_images;
  }
  if (IsApiMetal()) {
    return 32;
  }
  if (IsApiOpenCl()) {
    return 128;
  }
  return 1;
}

bool GpuInfo::IsApiOpenGl() const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_78(mht_78_v, 1358, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApiOpenGl");
 return gpu_api == GpuApi::kOpenGl; }

bool GpuInfo::IsApiOpenGl31OrAbove() const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_79(mht_79_v, 1363, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApiOpenGl31OrAbove");

  if (!IsApiOpenGl()) {
    return false;
  }
  return (opengl_info.major_version == 3 && opengl_info.minor_version >= 1) ||
         opengl_info.major_version > 3;
}

bool GpuInfo::IsApiVulkan() const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_80(mht_80_v, 1374, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApiVulkan");
 return gpu_api == GpuApi::kVulkan; }

bool GpuInfo::IsApiMetal() const {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_81(mht_81_v, 1379, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApiMetal");
 return gpu_api == GpuApi::kMetal; }

bool GpuInfo::IsApiOpenCl() const {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_82(mht_82_v, 1384, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsApiOpenCl");
 return gpu_api == GpuApi::kOpenCl; }

bool GpuInfo::IsGlsl() const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_83(mht_83_v, 1389, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsGlsl");
 return IsApiOpenGl() || IsApiVulkan(); }

bool GpuInfo::IsGlslSupportsExplicitFp16() const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_84(mht_84_v, 1394, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsGlslSupportsExplicitFp16");

  if (IsApiOpenGl() && opengl_info.SupportsExplicitFp16()) {
    return true;
  }
  if (IsApiVulkan() && vulkan_info.SupportsExplicitFp16()) {
    return true;
  }
  return false;
}

bool GpuInfo::IsCL11OrHigher() const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_85(mht_85_v, 1407, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsCL11OrHigher");

  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0;
}

bool GpuInfo::IsCL20OrHigher() const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_86(mht_86_v, 1417, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsCL20OrHigher");

  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0 &&
         opencl_info.cl_version != OpenClVersion::kCl1_1 &&
         opencl_info.cl_version != OpenClVersion::kCl1_2;
}

bool GpuInfo::IsCL30OrHigher() const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTcc mht_87(mht_87_v, 1429, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.cc", "GpuInfo::IsCL30OrHigher");

  if (!IsApiOpenCl()) {
    return false;
  }
  return IsCL20OrHigher() && opencl_info.cl_version != OpenClVersion::kCl2_0 &&
         opencl_info.cl_version != OpenClVersion::kCl2_1 &&
         opencl_info.cl_version != OpenClVersion::kCl2_2;
}

}  // namespace gpu
}  // namespace tflite
