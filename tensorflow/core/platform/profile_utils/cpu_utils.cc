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
class MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc() {
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

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

#include <fstream>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "absl/base/call_once.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

namespace tensorflow {
namespace profile_utils {

/* static */ constexpr int64_t CpuUtils::INVALID_FREQUENCY;

static ICpuUtilsHelper* cpu_utils_helper_instance_ = nullptr;

#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
/* static */ uint64 CpuUtils::GetCycleCounterFrequency() {
  static const uint64 cpu_frequency = GetCycleCounterFrequencyImpl();
  return cpu_frequency;
}
#else
/* static */ int64_t CpuUtils::GetCycleCounterFrequency() {
  static const int64_t cpu_frequency = GetCycleCounterFrequencyImpl();
  return cpu_frequency;
}
#endif

/* static */ double CpuUtils::GetMicroSecPerClock() {
  static const double micro_sec_per_clock =
      (1000.0 * 1000.0) / static_cast<double>(GetCycleCounterFrequency());
  return micro_sec_per_clock;
}

/* static */ void CpuUtils::ResetClockCycle() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::ResetClockCycle");

  GetCpuUtilsHelperSingletonInstance().ResetClockCycle();
}

/* static */ void CpuUtils::EnableClockCycleProfiling() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::EnableClockCycleProfiling");

  GetCpuUtilsHelperSingletonInstance().EnableClockCycleProfiling();
}

/* static */ void CpuUtils::DisableClockCycleProfiling() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::DisableClockCycleProfiling");

  GetCpuUtilsHelperSingletonInstance().DisableClockCycleProfiling();
}

/* static */ std::chrono::duration<double> CpuUtils::ConvertClockCycleToTime(
    const int64_t clock_cycle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::ConvertClockCycleToTime");

  return std::chrono::duration<double>(static_cast<double>(clock_cycle) /
                                       GetCycleCounterFrequency());
}

/* static */ int64_t CpuUtils::GetCycleCounterFrequencyImpl() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::GetCycleCounterFrequencyImpl");

// TODO(satok): do not switch by macro here
#if defined(__ANDROID__)
  return GetCpuUtilsHelperSingletonInstance().CalculateCpuFrequency();
#elif defined(__linux__)
  // Read the contents of /proc/cpuinfo.
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo) {
    LOG(WARNING) << "Failed to open /proc/cpuinfo";
    return INVALID_FREQUENCY;
  }
  string line;
  while (std::getline(cpuinfo, line)) {
    double cpu_freq = 0.0;
    int retval = 0;
    double freq_factor = 2.0;
#if (defined(__powerpc__) || \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
    retval = sscanf(line.c_str(), "clock              : %lfMHz", &cpu_freq);
    freq_factor = 1.0;
#elif defined(__s390x__)
    retval = sscanf(line.c_str(), "bogomips per cpu: %lf", &cpu_freq);
#elif defined(__aarch64__)
    retval = sscanf(line.c_str(), "BogoMIPS : %lf", &cpu_freq);
#else
    retval = sscanf(line.c_str(), "bogomips : %lf", &cpu_freq);
#endif
    if (retval > 0) {
      const double freq_ghz = cpu_freq / 1000.0 / freq_factor;
      if (retval != 1 || freq_ghz < 0.01) {
        LOG(WARNING) << "Failed to get CPU frequency: " << freq_ghz << " GHz";
        return INVALID_FREQUENCY;
      }
      const int64_t freq_n =
          static_cast<int64_t>(freq_ghz * 1000.0 * 1000.0 * 1000.0);
      VLOG(1) << "CPU Frequency: " << freq_n << " Hz";
      return freq_n;
    }
  }
  LOG(WARNING)
      << "Failed to find bogomips or clock in /proc/cpuinfo; cannot determine "
         "CPU frequency";
  return INVALID_FREQUENCY;
#elif defined(__APPLE__)
  int64 freq_hz = 0;
  size_t freq_size = sizeof(freq_hz);
  int retval =
      sysctlbyname("hw.cpufrequency_max", &freq_hz, &freq_size, NULL, 0);
  if (retval != 0 || freq_hz < 1e6) {
    LOG(WARNING) << "Failed to get CPU frequency: " << freq_hz << " Hz";
    return INVALID_FREQUENCY;
  }
  return freq_hz;
#elif defined(_WIN32)
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart;
#else
  // TODO(satok): Support other OS if needed
  // Return INVALID_FREQUENCY on unsupported OS
  return INVALID_FREQUENCY;
#endif
}

/* static */ ICpuUtilsHelper& CpuUtils::GetCpuUtilsHelperSingletonInstance() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTcc mht_5(mht_5_v, 327, "", "./tensorflow/core/platform/profile_utils/cpu_utils.cc", "CpuUtils::GetCpuUtilsHelperSingletonInstance");

  static absl::once_flag flag;
  absl::call_once(flag, []() {
    if (cpu_utils_helper_instance_ != nullptr) {
      LOG(FATAL) << "cpu_utils_helper_instance_ is already instantiated.";
    }
#if defined(__ANDROID__) && (__ANDROID_API__ >= 21) && \
    (defined(__ARM_ARCH_7A__) || defined(__aarch64__))
    cpu_utils_helper_instance_ = new AndroidArmV7ACpuUtilsHelper();
#else
      cpu_utils_helper_instance_ = new DefaultCpuUtilsHelper();
#endif
  });
  return *cpu_utils_helper_instance_;
}

}  // namespace profile_utils
}  // namespace tensorflow
