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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorder_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorder_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorder_testDTcc() {
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
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"

#include <atomic>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::SpinForNanos;
using ::testing::ElementsAre;

MATCHER_P(Named, name, "") { return arg.name == name; }

TEST(RecorderTest, SingleThreaded) {
  int64_t start_time = GetCurrentTimeNanos();
  int64_t end_time = start_time + UniToNano(1);

  TraceMeRecorder::Record({"before", start_time, end_time});
  TraceMeRecorder::Start(/*level=*/1);
  TraceMeRecorder::Record({"during1", start_time, end_time});
  TraceMeRecorder::Record({"during2", start_time, end_time});
  auto results = TraceMeRecorder::Stop();
  TraceMeRecorder::Record({"after", start_time, end_time});

  ASSERT_EQ(results.size(), 1);
  EXPECT_THAT(results[0].events,
              ElementsAre(Named("during1"), Named("during2")));
}

// Checks the functional behavior of the recorder, when used from several
// unsynchronized threads.
//
// Each thread records a stream of events.
//   Thread 0: activity=0, activity=1, activity=2, ...
//   Thread 1: activity=0, activity=1, activity=2, ...
//   ...
//
// We turn the recorder on and off repeatedly in sessions, expecting to see:
//   - data from every thread (eventually - maybe not every session)
//   - unbroken sessions: a consecutive sequence of IDs from each thread
//   - gaps between sessions: a thread's IDs should be non-consecutive overall
TEST(RecorderTest, Multithreaded) {
  constexpr static int kNumThreads = 4;

  // Start several threads writing events.
  tensorflow::Notification start;
  tensorflow::Notification stop;
  thread::ThreadPool pool(Env::Default(), "testpool", kNumThreads);
  std::atomic<int> thread_count = {0};
  for (int i = 0; i < kNumThreads; i++) {
    pool.Schedule([&start, &stop, &thread_count] {
      uint64 j = 0;
      bool was_active = false;
      auto record_event = [&j]() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorder_testDTcc mht_0(mht_0_v, 253, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder_test.cc", "lambda");

        int64_t start_time = GetCurrentTimeNanos();
        int64_t end_time = start_time + UniToNano(1);
        TraceMeRecorder::Record(
            {/*name=*/absl::StrCat(j++), start_time, end_time});
      };
      thread_count.fetch_add(1, std::memory_order_relaxed);
      start.WaitForNotification();
      while (!stop.HasBeenNotified()) {
        // Mimicking production usage, we guard with a racy check.
        // In principle this isn't needed, but a feedback loop can form:
        // 1) many events accumulate while the recorder is off
        // 2) clearing/analyzing these events is slow
        // 3) while clearing, more events are accumulating, causing 1
        if (TraceMeRecorder::Active()) {
          record_event();
          was_active = true;
        }
        // Record some events after the recorder is no longer active to simulate
        // point 1 and 3.
        if (was_active && !TraceMeRecorder::Active()) {
          record_event();
          record_event();
          was_active = false;
        }
        // This snowballs into OOM in some configurations, causing flakiness.
        // Keep this big enough to prevent OOM and small enough such that
        // each thread records at least one event.
        SpinForNanos(10);
      }
    });
  }

  // For each thread, keep track of which events we've seen.
  struct ThreadState {
    bool split_session = false;
    bool overlapping_sessions = false;
    std::set<uint64> events;
  };
  absl::flat_hash_map<uint32 /*tid*/, ThreadState> thread_state;
  // We expect each thread to eventually have multiple events, not all in a
  // contiguous range.
  auto done = [&thread_state] {
    for (const auto& id_and_thread : thread_state) {
      auto& t = id_and_thread.second;
      if (t.events.size() < 2) return false;
    }
    return true;
  };

  // Wait while all the threads are spun up.
  while (thread_count.load(std::memory_order_relaxed) < kNumThreads) {
    LOG(INFO) << "Waiting for all threads to spin up...";
    SleepForMillis(1);
  }

  // We will probably be done after two iterations (with each thread getting
  // some events each iteration). No guarantees as all the threads might not get
  // scheduled in a session, so try for a while.
  start.Notify();
  constexpr static int kMaxIters = 100;
  for (int iters = 0; iters < kMaxIters && !done(); ++iters) {
    LOG(INFO) << "Looping until convergence, iteration: " << iters;
    TraceMeRecorder::Start(/*level=*/1);
    SleepForMillis(100);
    auto results = TraceMeRecorder::Stop();
    for (const auto& thread : results) {
      if (thread.events.empty()) continue;
      auto& state = thread_state[thread.thread.tid];

      std::set<uint64> session_events;
      uint64 current = 0;
      for (const auto& event : thread.events) {
        uint64 activity_id;
        ASSERT_TRUE(absl::SimpleAtoi(event.name, &activity_id));
        session_events.emplace(activity_id);
        // Session events should be contiguous.
        if (current != 0 && activity_id != current + 1) {
          state.split_session = true;
        }
        current = activity_id;
      }

      for (const auto& event : session_events) {
        auto result = state.events.emplace(event);
        if (!result.second) {
          // Session events should not overlap with those from previous
          // sessions.
          state.overlapping_sessions = true;
        }
      }
    }
    SleepForMillis(1);
  }
  stop.Notify();

  for (const auto& id_and_thread : thread_state) {
    auto& thread = id_and_thread.second;
    EXPECT_FALSE(thread.split_session)
        << "Expected contiguous events in a session";
    EXPECT_FALSE(thread.overlapping_sessions) << "Expected disjoint sessions";
    EXPECT_GT(thread.events.size(), 1)
        << "Expected gaps in thread events between sessions";
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
