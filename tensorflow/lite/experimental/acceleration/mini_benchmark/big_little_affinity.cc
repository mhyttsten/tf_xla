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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSbig_little_affinityDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSbig_little_affinityDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSbig_little_affinityDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>

#include "include/cpuinfo.h"

namespace tflite {
namespace acceleration {

namespace {
bool IsInOrderArch(cpuinfo_uarch arch) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSbig_little_affinityDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.cc", "IsInOrderArch");

  switch (arch) {
    case cpuinfo_uarch_cortex_a53:
    case cpuinfo_uarch_cortex_a55r0:
    case cpuinfo_uarch_cortex_a55:
    case cpuinfo_uarch_cortex_a57:
      return true;
    default:
      return false;
  }
  return false;
}
}  // namespace

BigLittleAffinity GetAffinity() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSbig_little_affinityDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.cc", "GetAffinity");

  BigLittleAffinity affinity;
  if (!cpuinfo_initialize()) {
    return affinity;
  }
  std::map<uint32_t, uint64_t> cluster_to_max_frequency;
  uint64_t smallest_max_frequency = UINT64_MAX;
  uint64_t largest_max_frequency = 0;
  uint64_t processors_count = cpuinfo_get_processors_count();
  for (auto i = 0; i < processors_count; i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    if (processor->core->frequency > 0) {
      cluster_to_max_frequency[processor->cluster->cluster_id] =
          processor->core->frequency;
      smallest_max_frequency =
          std::min(smallest_max_frequency, processor->core->frequency);
      largest_max_frequency =
          std::max(largest_max_frequency, processor->core->frequency);
    }
  }

  int count_of_processors_with_largest_max_frequency = 0;
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    uint64_t max_frequency =
        cluster_to_max_frequency[processor->cluster->cluster_id];
    if (max_frequency == largest_max_frequency) {
      ++count_of_processors_with_largest_max_frequency;
    }
  }
  std::set<cpuinfo_uarch> archs;

  // Three variants for detecting the big/little split:
  // - all cores have the same frequency, check the uarch for in-order (on
  //   big.LITTLE, the big cores are typically out-of-order and the LITTLE
  //   cores in-order)
  // - if there are 2 cores with largest max frequency, those are counted as big
  // - otherwise the cores with smallest max frequency are counted as LITTLE
  for (auto i = 0; i < cpuinfo_get_processors_count(); i++) {
    const struct cpuinfo_processor* processor = cpuinfo_get_processor(i);
    uint64_t max_frequency =
        cluster_to_max_frequency[processor->cluster->cluster_id];
    bool is_little;
    archs.insert(processor->core->uarch);
    if (count_of_processors_with_largest_max_frequency ==
        cpuinfo_get_processors_count()) {
      is_little = IsInOrderArch(processor->core->uarch);
    } else if (count_of_processors_with_largest_max_frequency == 2) {
      is_little = (max_frequency != largest_max_frequency);
    } else {
      is_little = (max_frequency == smallest_max_frequency);
    }
#ifdef __ANDROID__
    // On desktop linux there are easily more processors than bits in an int, so
    // skip this code. It's still convenient to enable the rest of the code on
    // non-Android for quicker testing.
    if (is_little) {
      affinity.little_core_affinity |= (0x1 << processor->linux_id);
    } else {
      affinity.big_core_affinity |= (0x1 << processor->linux_id);
    }
#endif  // __ANDROID__
  }
  // After the detection we may have determined that all cores are big or
  // LITTLE. This is ok if there is only one cluster or if all the cores are the
  // same, and in that case we return the same for both masks.
  if (cluster_to_max_frequency.size() == 1) {
    // Only one cluster.
    affinity.big_core_affinity = affinity.little_core_affinity =
        std::max(affinity.big_core_affinity, affinity.little_core_affinity);
  } else if (count_of_processors_with_largest_max_frequency ==
                 cpuinfo_get_processors_count() &&
             archs.size() == 1) {
    // All cores have same uarch and frequency.
    affinity.big_core_affinity = affinity.little_core_affinity =
        std::max(affinity.big_core_affinity, affinity.little_core_affinity);
  }
  return affinity;
}

}  // namespace acceleration
}  // namespace tflite
