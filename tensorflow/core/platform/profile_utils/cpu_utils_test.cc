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
class MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utils_testDTcc() {
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
// This class is designed to get accurate profiles for programs

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/clock_cycle_profiler.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profile_utils {

static constexpr bool DBG = false;

class CpuUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprofile_utilsPScpu_utils_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/platform/profile_utils/cpu_utils_test.cc", "SetUp");
 CpuUtils::EnableClockCycleProfiling(); }
};

TEST_F(CpuUtilsTest, SetUpTestCase) {}

TEST_F(CpuUtilsTest, TearDownTestCase) {}

TEST_F(CpuUtilsTest, CheckGetCurrentClockCycle) {
  static constexpr int LOOP_COUNT = 10;
  const uint64 start_clock_count = CpuUtils::GetCurrentClockCycle();
  CHECK_GT(start_clock_count, 0);
  uint64 prev_clock_count = start_clock_count;
  for (int i = 0; i < LOOP_COUNT; ++i) {
    const uint64 clock_count = CpuUtils::GetCurrentClockCycle();
    CHECK_GE(clock_count, prev_clock_count);
    prev_clock_count = clock_count;
  }
  const uint64 end_clock_count = CpuUtils::GetCurrentClockCycle();
  if (DBG) {
    LOG(INFO) << "start clock = " << start_clock_count;
    LOG(INFO) << "end clock = " << end_clock_count;
    LOG(INFO) << "average clock = "
              << ((end_clock_count - start_clock_count) / LOOP_COUNT);
  }
}

TEST_F(CpuUtilsTest, CheckCycleCounterFrequency) {
#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
  const uint64 cpu_frequency = CpuUtils::GetCycleCounterFrequency();
  CHECK_GT(cpu_frequency, 0);
  CHECK_NE(cpu_frequency, unsigned(CpuUtils::INVALID_FREQUENCY));
#else
  const int64_t cpu_frequency = CpuUtils::GetCycleCounterFrequency();
  CHECK_GT(cpu_frequency, 0);
  CHECK_NE(cpu_frequency, CpuUtils::INVALID_FREQUENCY);
#endif
  if (DBG) {
    LOG(INFO) << "Cpu frequency = " << cpu_frequency;
  }
}

TEST_F(CpuUtilsTest, CheckMicroSecPerClock) {
  const double micro_sec_per_clock = CpuUtils::GetMicroSecPerClock();
  CHECK_GT(micro_sec_per_clock, 0.0);
  if (DBG) {
    LOG(INFO) << "Micro sec per clock = " << micro_sec_per_clock;
  }
}

TEST_F(CpuUtilsTest, SimpleUsageOfClockCycleProfiler) {
  static constexpr int LOOP_COUNT = 10;
  ClockCycleProfiler prof;
  for (int i = 0; i < LOOP_COUNT; ++i) {
    prof.Start();
    prof.Stop();
  }
  EXPECT_EQ(LOOP_COUNT, static_cast<int>(prof.GetCount() + 0.5));
  if (DBG) {
    prof.DumpStatistics("CpuUtilsTest");
  }
}

}  // namespace profile_utils
}  // namespace tensorflow
