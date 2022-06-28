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
class MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc() {
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

#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

#if defined(__ANDROID__) && (__ANDROID_API__ >= 21) && \
    (defined(__ARM_ARCH_7A__) || defined(__aarch64__))

#include <asm/unistd.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace profile_utils {

/* static */ constexpr int AndroidArmV7ACpuUtilsHelper::INVALID_FD;
/* static */ constexpr int64 AndroidArmV7ACpuUtilsHelper::INVALID_CPU_FREQUENCY;

void AndroidArmV7ACpuUtilsHelper::ResetClockCycle() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::ResetClockCycle");

  if (!is_initialized_) {
    return;
  }
  ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
}

uint64 AndroidArmV7ACpuUtilsHelper::GetCurrentClockCycle() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::GetCurrentClockCycle");

  if (!is_initialized_) {
    return 1;  // DUMMY
  }
  long long count;
  read(fd_, &count, sizeof(long long));
  return static_cast<uint64>(count);
}

void AndroidArmV7ACpuUtilsHelper::EnableClockCycleProfiling() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::EnableClockCycleProfiling");

  if (!is_initialized_) {
    // Initialize here to avoid unnecessary initialization
    InitializeInternal();
  }
    const int64 cpu0_scaling_min = ReadCpuFrequencyFile(0, "scaling_min");
    const int64 cpu0_scaling_max = ReadCpuFrequencyFile(0, "scaling_max");
    if (cpu0_scaling_max != cpu0_scaling_min) {
      LOG(WARNING) << "You enabled clock cycle profile but frequency may "
                   << "be scaled. (max = " << cpu0_scaling_max << ", min "
                   << cpu0_scaling_min << ")";
    }
    ResetClockCycle();
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
}

void AndroidArmV7ACpuUtilsHelper::DisableClockCycleProfiling() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::DisableClockCycleProfiling");

  if (!is_initialized_) {
    // Initialize here to avoid unnecessary initialization
    InitializeInternal();
  }
  ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
}

int64 AndroidArmV7ACpuUtilsHelper::CalculateCpuFrequency() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::CalculateCpuFrequency");

  return ReadCpuFrequencyFile(0, "scaling_cur");
}

void AndroidArmV7ACpuUtilsHelper::InitializeInternal() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::InitializeInternal");

  perf_event_attr pe;

  memset(&pe, 0, sizeof(perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(perf_event_attr);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;

  fd_ = OpenPerfEvent(&pe, 0, -1, -1, 0);
  if (fd_ == INVALID_FD) {
    LOG(WARNING) << "Error opening perf event";
    is_initialized_ = false;
  } else {
    is_initialized_ = true;
  }
}

int AndroidArmV7ACpuUtilsHelper::OpenPerfEvent(perf_event_attr *const hw_event,
                                               const pid_t pid, const int cpu,
                                               const int group_fd,
                                               const unsigned long flags) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_6(mht_6_v, 295, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::OpenPerfEvent");

  const int ret =
      syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

int64 AndroidArmV7ACpuUtilsHelper::ReadCpuFrequencyFile(
    const int cpu_id, const char *const type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSandroid_armv7a_cpu_utils_helperDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.cc", "AndroidArmV7ACpuUtilsHelper::ReadCpuFrequencyFile");

  const string file_path = strings::Printf(
      "/sys/devices/system/cpu/cpu%d/cpufreq/%s_freq", cpu_id, type);
  FILE *fp = fopen(file_path.c_str(), "r");
  if (fp == nullptr) {
    return INVALID_CPU_FREQUENCY;
  }
  int64_t freq_in_khz = INVALID_CPU_FREQUENCY;
  const int retval = fscanf(fp, "%" SCNd64, &freq_in_khz);
  if (retval < 0) {
    LOG(WARNING) << "Failed to \"" << file_path << "\"";
    if (fclose(fp) != 0) {
      LOG(WARNING) << "fclose() failed: " << strerror(errno);
    }
    return INVALID_CPU_FREQUENCY;
  }
  if (fclose(fp) != 0) {
    LOG(WARNING) << "fclose() failed: " << strerror(errno);
  }
  return freq_in_khz * 1000;  // The file contains cpu frequency in khz
}

}  // namespace profile_utils
}  // namespace tensorflow

#endif  // defined(__ANDROID__) && (__ANDROID_API__ >= 21) &&
        // (defined(__ARM_ARCH_7A__) || defined(__aarch64__))
