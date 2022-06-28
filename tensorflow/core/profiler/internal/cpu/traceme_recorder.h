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
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh() {
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


#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace internal {

// Current trace level.
// Static atomic so TraceMeRecorder::Active can be fast and non-blocking.
// Modified by TraceMeRecorder singleton when tracing starts/stops.
TF_EXPORT extern std::atomic<int> g_trace_level;

}  // namespace internal

// TraceMeRecorder is a singleton repository of TraceMe events.
// It can be safely and cheaply appended to by multiple threads.
//
// Start() and Stop() must be called in pairs, Stop() returns the events added
// since the previous Start().
//
// This is the backend for TraceMe instrumentation.
// The profiler starts the recorder, the TraceMe destructor records complete
// events. TraceMe::ActivityStart records start events, and TraceMe::ActivityEnd
// records end events. The profiler then stops the recorder and finds start/end
// pairs. (Unpaired start/end events are discarded at that point).
class TraceMeRecorder {
 public:
  // An Event is either the start of a TraceMe, the end of a TraceMe, or both.
  // Times are in ns since the Unix epoch.
  // A negative time encodes the activity_id used to pair up the start of an
  // event with its end.
  struct Event {
    bool IsComplete() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_0(mht_0_v, 228, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "IsComplete");
 return start_time > 0 && end_time > 0; }
    bool IsStart() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "IsStart");
 return end_time < 0; }
    bool IsEnd() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_2(mht_2_v, 236, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "IsEnd");
 return start_time < 0; }

    int64_t ActivityId() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_3(mht_3_v, 241, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "ActivityId");

      if (IsStart()) return -end_time;
      if (IsEnd()) return -start_time;
      return 1;  // complete
    }

    std::string name;
    int64_t start_time;
    int64_t end_time;
  };
  struct ThreadInfo {
    uint32 tid;
    std::string name;
  };
  struct ThreadEvents {
    ThreadInfo thread;
    std::deque<Event> events;
  };
  using Events = std::vector<ThreadEvents>;

  // Starts recording of TraceMe().
  // Only traces <= level will be recorded.
  // Level must be >= 0. If level is 0, no traces will be recorded.
  static bool Start(int level) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_4(mht_4_v, 267, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "Start");
 return Get()->StartRecording(level); }

  // Stops recording and returns events recorded since Start().
  // Events passed to Record after Stop has started will be dropped.
  static Events Stop() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_5(mht_5_v, 274, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "Stop");
 return Get()->StopRecording(); }

  // Returns whether we're currently recording. Racy, but cheap!
  static inline bool Active(int level = 1) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTh mht_6(mht_6_v, 280, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.h", "Active");

    return internal::g_trace_level.load(std::memory_order_acquire) >= level;
  }

  // Default value for trace_level_ when tracing is disabled
  static constexpr int kTracingDisabled = -1;

  // Records an event. Non-blocking.
  static void Record(Event&& event);

  // Returns an activity_id for TraceMe::ActivityStart.
  static int64_t NewActivityId();

 private:
  class ThreadLocalRecorder;
  class ThreadLocalRecorderWrapper;

  // Returns singleton.
  static TraceMeRecorder* Get();

  TraceMeRecorder() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMeRecorder);

  void RegisterThread(uint32 tid, std::shared_ptr<ThreadLocalRecorder> thread);
  void UnregisterThread(uint32 tid);

  bool StartRecording(int level);
  Events StopRecording();

  // Clears events from all active threads that were added due to Record
  // racing with StopRecording.
  void Clear() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Gathers events from all active threads, and clears their buffers.
  TF_MUST_USE_RESULT Events Consume() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutex mutex_;
  // A ThreadLocalRecorder stores trace events. Ownership is shared with
  // ThreadLocalRecorderWrapper, which is allocated in thread_local storage.
  // ThreadLocalRecorderWrapper creates the ThreadLocalRecorder and registers it
  // with TraceMeRecorder on the first TraceMe executed on a thread while
  // tracing is active. If the thread is destroyed during tracing, the
  // ThreadLocalRecorder is marked inactive but remains alive until tracing
  // stops so the events can be retrieved.
  absl::flat_hash_map<uint32, std::shared_ptr<ThreadLocalRecorder>> threads_
      TF_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_TRACEME_RECORDER_H_
