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
class MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/profiler.h"

#include <unistd.h>

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <thread>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace profiling {
namespace {

double GetDurationOfEventMs(const ProfileEvent* event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/profiling/profiler_test.cc", "GetDurationOfEventMs");

  return (event->end_timestamp_us - event->begin_timestamp_us) / 1e3;
}

void SleepForQuarterSecond(tflite::Profiler* profiler) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/lite/profiling/profiler_test.cc", "SleepForQuarterSecond");

  ScopedProfile profile(profiler, "SleepForQuarter");
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

void ChildFunction(tflite::Profiler* profiler) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/lite/profiling/profiler_test.cc", "ChildFunction");

  ScopedProfile profile(profiler, "Child");
  SleepForQuarterSecond(profiler);
}

void ParentFunction(tflite::Profiler* profiler) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofiler_testDTcc mht_3(mht_3_v, 222, "", "./tensorflow/lite/profiling/profiler_test.cc", "ParentFunction");

  ScopedProfile profile(profiler, "Parent");
  for (int i = 0; i < 2; i++) {
    ChildFunction(profiler);
  }
}

TEST(ProfilerTest, NoProfilesAreCollectedWhenDisabled) {
  BufferedProfiler profiler(1024);
  ParentFunction(&profiler);
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(0, profile_events.size());
}

TEST(ProfilerTest, NoProfilesAreCollectedWhenEventTypeUnsupported) {
  BufferedProfiler profiler(1024);
  tflite::Profiler* p = &profiler;
  p->AddEvent("Hello",
              Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT,
              /*start*/ 0, /*end*/ 1,
              /*event_metadata*/ 2);
  auto handler = p->BeginEvent(
      "begin", Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT, 0);
  p->EndEvent(handler);
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(0, profile_events.size());
}

TEST(ProfilingTest, ProfilesAreCollected) {
  BufferedProfiler profiler(1024);
  profiler.StartProfiling();
  ParentFunction(&profiler);
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  // ParentFunction calls the ChildFunction 2 times.
  // Each ChildFunction calls SleepForQuarterSecond once.
  // We expect 1 entry for ParentFunction, 2 for ChildFunction and 2 for
  // SleepForQuarterSecond: Total: 1+ 2 + 2 = 5
  //  Profiles should look like:
  //  Parent ~ 500 ms (due to 2 Child calls)
  //   - Child ~ 250 ms (due to SleepForQuarter calls)
  //       - SleepForQuarter ~ 250ms
  //   - Child ~ 250 ms (due to SleepForQuarter calls)
  //      - SleepForQuarter ~ 250ms
  //
  ASSERT_EQ(5, profile_events.size());
  EXPECT_EQ("Parent", profile_events[0]->tag);
  EXPECT_EQ("Child", profile_events[1]->tag);
  EXPECT_EQ("SleepForQuarter", profile_events[2]->tag);
  EXPECT_EQ("Child", profile_events[3]->tag);
  EXPECT_EQ("SleepForQuarter", profile_events[4]->tag);

#ifndef ADDRESS_SANITIZER
  // ASAN build is sometimes very slow. Set a large epsilon to avoid flakiness.
  // Due to flakiness, just verify relative values match.
  const int eps_ms = 50;
  auto parent_ms = GetDurationOfEventMs(profile_events[0]);
  double child_ms[2], sleep_for_quarter_ms[2];
  child_ms[0] = GetDurationOfEventMs(profile_events[1]);
  child_ms[1] = GetDurationOfEventMs(profile_events[3]);
  sleep_for_quarter_ms[0] = GetDurationOfEventMs(profile_events[2]);
  sleep_for_quarter_ms[1] = GetDurationOfEventMs(profile_events[4]);
  EXPECT_NEAR(parent_ms, child_ms[0] + child_ms[1], eps_ms);
  EXPECT_NEAR(child_ms[0], sleep_for_quarter_ms[0], eps_ms);
  EXPECT_NEAR(child_ms[1], sleep_for_quarter_ms[1], eps_ms);
#endif
}

TEST(ProfilingTest, NullProfiler) {
  Profiler* profiler = nullptr;
  { SCOPED_TAGGED_OPERATOR_PROFILE(profiler, "noop", 1); }
}

TEST(ProfilingTest, ScopedProfile) {
  BufferedProfiler profiler(1024);
  profiler.StartProfiling();
  { SCOPED_TAGGED_OPERATOR_PROFILE(&profiler, "noop", 1); }
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(1, profile_events.size());
}

TEST(ProfilingTest, NoopProfiler) {
  NoopProfiler profiler;
  profiler.StartProfiling();
  { SCOPED_TAGGED_OPERATOR_PROFILE(&profiler, "noop", 1); }
  profiler.StopProfiling();
  auto profile_events = profiler.GetProfileEvents();
  EXPECT_EQ(0, profile_events.size());
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
