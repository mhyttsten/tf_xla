/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_
#define TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_
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
class MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh() {
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


#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace tflite {
namespace profiling {
namespace memory {

// This class could help to tell the peak memory footprint of a running program.
// It achieves this by spawning a thread to check the memory usage periodically
// at a pre-defined frequency.
class MemoryUsageMonitor {
 public:
  // A helper class that does memory usage sampling. This allows injecting an
  // external dependency for the sake of testing or providing platform-specific
  // implementations.
  class Sampler {
   public:
    virtual ~Sampler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_0(mht_0_v, 210, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "~Sampler");
}
    virtual bool IsSupported() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_1(mht_1_v, 214, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "IsSupported");
 return MemoryUsage::IsSupported(); }
    virtual MemoryUsage GetMemoryUsage() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_2(mht_2_v, 218, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "GetMemoryUsage");

      return tflite::profiling::memory::GetMemoryUsage();
    }
    virtual void SleepFor(const absl::Duration& duration) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_3(mht_3_v, 224, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "SleepFor");

      absl::SleepFor(duration);
    }
  };

  static constexpr float kInvalidMemUsageMB = -1.0f;

  explicit MemoryUsageMonitor(int sampling_interval_ms = 50)
      : MemoryUsageMonitor(sampling_interval_ms,
                           std::unique_ptr<Sampler>(new Sampler())) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_4(mht_4_v, 236, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "MemoryUsageMonitor");
}
  MemoryUsageMonitor(int sampling_interval_ms,
                     std::unique_ptr<Sampler> sampler);
  ~MemoryUsageMonitor() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_5(mht_5_v, 242, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "~MemoryUsageMonitor");
 StopInternal(); }

  void Start();
  void Stop();

  // For simplicity, we will return kInvalidMemUsageMB for the either following
  // conditions:
  // 1. getting memory usage isn't supported on the platform.
  // 2. the memory usage is being monitored (i.e. we've created the
  // 'check_memory_thd_'.
  float GetPeakMemUsageInMB() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSmemory_usage_monitorDTh mht_6(mht_6_v, 255, "", "./tensorflow/lite/profiling/memory_usage_monitor.h", "GetPeakMemUsageInMB");

    if (!is_supported_ || check_memory_thd_ != nullptr) {
      return kInvalidMemUsageMB;
    }
    return peak_max_rss_kb_ / 1024.0;
  }

  MemoryUsageMonitor(MemoryUsageMonitor&) = delete;
  MemoryUsageMonitor& operator=(const MemoryUsageMonitor&) = delete;
  MemoryUsageMonitor(MemoryUsageMonitor&&) = delete;
  MemoryUsageMonitor& operator=(const MemoryUsageMonitor&&) = delete;

 private:
  void StopInternal();

  std::unique_ptr<Sampler> sampler_ = nullptr;
  bool is_supported_ = false;
  std::unique_ptr<absl::Notification> stop_signal_ = nullptr;
  absl::Duration sampling_interval_;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
  int64_t peak_max_rss_kb_ = static_cast<int64_t>(kInvalidMemUsageMB * 1024);
};

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_MEMORY_USAGE_MONITOR_H_
