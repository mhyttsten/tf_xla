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

#ifndef TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
#define TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh() {
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


#include <algorithm>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

namespace tensorflow {

class ClockCycleProfiler {
 public:
  ClockCycleProfiler() = default;

  // Start counting clock cycle.
  inline void Start() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "Start");

    CHECK(!IsStarted()) << "Profiler has been already started.";
    start_clock_ = GetCurrentClockCycleInternal();
  }

  // Stop counting clock cycle.
  inline void Stop() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_1(mht_1_v, 210, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "Stop");

    CHECK(IsStarted()) << "Profiler is not started yet.";
    AccumulateClockCycle();
  }

  // Get how many times Start() is called.
  inline double GetCount() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_2(mht_2_v, 219, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "GetCount");

    CHECK(!IsStarted());
    return count_;
  }

  // Get average clock cycle.
  inline double GetAverageClockCycle() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_3(mht_3_v, 228, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "GetAverageClockCycle");

    CHECK(!IsStarted());
    return average_clock_cycle_;
  }

  // TODO(satok): Support more statistics (e.g. standard deviation)
  // Get worst clock cycle.
  inline double GetWorstClockCycle() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_4(mht_4_v, 238, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "GetWorstClockCycle");

    CHECK(!IsStarted());
    return worst_clock_cycle_;
  }

  // Dump statistics
  void DumpStatistics(const string& tag);

 private:
  inline uint64 GetCurrentClockCycleInternal() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_5(mht_5_v, 250, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "GetCurrentClockCycleInternal");

    const uint64 clockCycle = profile_utils::CpuUtils::GetCurrentClockCycle();
    if (clockCycle <= 0) {
      if (valid_) {
        LOG(WARNING) << "GetCurrentClockCycle is not implemented."
                     << " Return 1 instead.";
        valid_ = false;
      }
      return 1;
    } else {
      return clockCycle;
    }
  }

  inline bool IsStarted() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_6(mht_6_v, 267, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "IsStarted");
 return start_clock_ > 0; }

  inline void AccumulateClockCycle() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPSclock_cycle_profilerDTh mht_7(mht_7_v, 272, "", "./tensorflow/core/platform/profile_utils/clock_cycle_profiler.h", "AccumulateClockCycle");

    const uint64 now = GetCurrentClockCycleInternal();
    const double clock_diff = static_cast<double>(now - start_clock_);
    const double next_count = count_ + 1.0;
    const double next_count_inv = 1.0 / next_count;
    const double next_ave_cpu_clock =
        next_count_inv * (average_clock_cycle_ * count_ + clock_diff);
    count_ = next_count;
    average_clock_cycle_ = next_ave_cpu_clock;
    worst_clock_cycle_ = std::max(worst_clock_cycle_, clock_diff);
    start_clock_ = 0;
  }

  uint64 start_clock_{0};
  double count_{0.0};
  double average_clock_cycle_{0.0};
  double worst_clock_cycle_{0.0};
  bool valid_{true};

  TF_DISALLOW_COPY_AND_ASSIGN(ClockCycleProfiler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
