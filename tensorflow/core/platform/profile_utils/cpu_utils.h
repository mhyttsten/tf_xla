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
// This class is designed to get accurate profile for programs.

#ifndef TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
#define TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh() {
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


#include <chrono>
#include <memory>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/i_cpu_utils_helper.h"
#include "tensorflow/core/platform/types.h"

#if defined(ARMV6) || defined(__ARM_ARCH_7A__)
#include <sys/time.h>
#endif

#if defined(_WIN32)
#include <intrin.h>
#endif

namespace tensorflow {

namespace profile_utils {

// CpuUtils is a profiling tool with static functions
// designed to be called from multiple classes.
// A dedicated class which inherits ICpuUtilsHelper is
// stored as a function-local static variable which inherits
// GetCpuUtilsHelperSingletonInstance that caches CPU information,
// because loading CPU information may take a long time.
// Users must call EnableClockCycleProfiling before using CpuUtils.
class CpuUtils {
 public:
  // Constant for invalid frequency.
  // This value is returned when the frequency is not obtained somehow.
  static constexpr int64_t INVALID_FREQUENCY = -1;
  static constexpr uint64 DUMMY_CYCLE_CLOCK = 1;

  // Return current clock cycle. This function is designed to
  // minimize the overhead to get clock and maximize the accuracy of
  // time for profile.
  // This returns unsigned int because there is no guarantee that rdtsc
  // is less than 2 ^ 61.
  static inline uint64 GetCurrentClockCycle() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "GetCurrentClockCycle");

#if defined(__ANDROID__)
    return GetCpuUtilsHelperSingletonInstance().GetCurrentClockCycle();
// ----------------------------------------------------------------
#elif defined(_WIN32)
    return __rdtsc();
// ----------------------------------------------------------------
#elif defined(__x86_64__) || defined(__amd64__)
    uint64_t high, low;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (high << 32) | low;
// ----------------------------------------------------------------
#elif defined(__aarch64__)
    // System timer of ARMv8 runs at a different frequency than the CPU's.
    // The frequency is fixed, typically in the range 1-50MHz.  It can because
    // read at CNTFRQ special register.  We assume the OS has set up
    // the virtual timer properly.
    uint64_t virtual_timer_value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
    return virtual_timer_value;
// ----------------------------------------------------------------
// V6 is the earliest arm that has a standard cyclecount
#elif defined(ARMV6) || defined(__ARM_ARCH_7A__)
    uint32_t pmccntr;
    uint32_t pmuseren;
    uint32_t pmcntenset;
    // Read the user mode perf monitor counter access permissions.
    asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
    if (pmuseren & 1) {  // Allows reading perfmon counters for user mode code.
      asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
      if (pmcntenset & 0x80000000ul) {  // Is it counting?
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
        // The counter is set up to count every 64th cyclecount
        return static_cast<uint64>(pmccntr) * 64;  // Should optimize to << 64
      }
    }
    // Returning dummy clock when can't access to the counter
    return DUMMY_CYCLE_CLOCK;
#elif defined(__powerpc64__) || defined(__ppc64__)
    uint64 __t;
    __asm__ __volatile__("mfspr %0,268" : "=r"(__t));
    return __t;

#elif defined(__powerpc__) || defined(__ppc__)
    uint64 upper, lower, tmp;
    __asm__ volatile(
        "0:                     \n"
        "\tmftbu   %0           \n"
        "\tmftb    %1           \n"
        "\tmftbu   %2           \n"
        "\tcmpw    %2,%0        \n"
        "\tbne     0b           \n"
        : "=r"(upper), "=r"(lower), "=r"(tmp));
    return ((static_cast<uint64>(upper) << 32) | lower);
#elif defined(__s390x__)
    // TOD Clock of s390x runs at a different frequency than the CPU's.
    // The stepping is 244 picoseconds (~4Ghz).
    uint64 t;
    __asm__ __volatile__("stckf %0" : "=Q"(t));
    return t;
#else
    // TODO(satok): Support generic way to emulate clock count.
    // TODO(satok): Support other architectures if wanted.
    // Returning dummy clock when can't access to the counter
    return DUMMY_CYCLE_CLOCK;
#endif
  }

// Return cycle counter frequency.
// As this method caches the cpu frequency internally,
// the first call will incur overhead, but not subsequent calls.
#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
  static uint64 GetCycleCounterFrequency();
#else
  static int64_t GetCycleCounterFrequency();
#endif

  // Return micro second per each clock
  // As this method caches the cpu frequency internally,
  // the first call will incur overhead, but not subsequent calls.
  static double GetMicroSecPerClock();

  // Reset clock cycle
  // Resetting clock cycle is recommended to prevent
  // clock cycle counters from overflowing on some platforms.
  static void ResetClockCycle();

  // Enable/Disable clock cycle profile
  // You can enable / disable profile if it's supported by the platform
  static void EnableClockCycleProfiling();
  static void DisableClockCycleProfiling();

  // Return chrono::duration per each clock
  static std::chrono::duration<double> ConvertClockCycleToTime(
      const int64_t clock_cycle);

 private:
  class DefaultCpuUtilsHelper : public ICpuUtilsHelper {
   public:
    DefaultCpuUtilsHelper() = default;
    void ResetClockCycle() final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_1(mht_1_v, 332, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "ResetClockCycle");
}
    uint64 GetCurrentClockCycle() final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_2(mht_2_v, 336, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "GetCurrentClockCycle");
 return DUMMY_CYCLE_CLOCK; }
    void EnableClockCycleProfiling() final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_3(mht_3_v, 340, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "EnableClockCycleProfiling");
}
    void DisableClockCycleProfiling() final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_4(mht_4_v, 344, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "DisableClockCycleProfiling");
}
    int64_t CalculateCpuFrequency() final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utilsDTh mht_5(mht_5_v, 348, "", "./tensorflow/core/platform/profile_utils/cpu_utils.h", "CalculateCpuFrequency");
 return INVALID_FREQUENCY; }

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(DefaultCpuUtilsHelper);
  };

  // Return cpu frequency.
  // CAVEAT: as this method calls system call and parse the message,
  // this call may be slow. This is why this class caches the value by
  // StaticVariableInitializer.
  static int64_t GetCycleCounterFrequencyImpl();

  // Return a singleton of ICpuUtilsHelper
  // ICpuUtilsHelper is declared as a function-local static variable
  // for the following two reasons:
  // 1. Avoid passing instances to all classes which want
  // to use profiling tools in CpuUtils
  // 2. Minimize the overhead of acquiring ICpuUtilsHelper
  static ICpuUtilsHelper& GetCpuUtilsHelperSingletonInstance();

  TF_DISALLOW_COPY_AND_ASSIGN(CpuUtils);
};

}  // namespace profile_utils

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
