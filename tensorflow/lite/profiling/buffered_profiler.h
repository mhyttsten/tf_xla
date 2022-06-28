/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_
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
class MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh() {
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


#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace profiling {

// Controls whether profiling is enabled or disabled and collects profiles.
// TFLite is used on platforms that don't have posix threads, so the profiler is
// kept as simple as possible. It is designed to be used only on a single
// thread.
//
// Profiles are collected using Scoped*Profile objects that begin and end a
// profile event.
// An example usage is shown in the example below:
//
// Say Worker class has a DoWork method and we are interested in profiling
// the overall execution time for DoWork and time spent in Task1 and Task2
// functions.
//
// class Worker {
//  public:
//   void DoWork() {
//    ScopedProfile(&controller, "DoWork");
//    Task1();
//    Task2();
//    .....
//   }
//
//   void Task1() {
//    ScopedProfile(&controller, "Task1");
//    ....
//   }
//
//   void Task2() {
//    ScopedProfile(&controller, "Task2");
//   }
//
//    Profiler profiler;
// }
//
// We instrument the functions that need to be profiled.
//
// Profile can be collected by enable profiling and then getting profile
// events.
//
//  void ProfileWorker() {
//    Worker worker;
//    worker.profiler.EnableProfiling();
//    worker.DoWork();
//    worker.profiler.DisableProfiling();
//    // Profiling is complete, extract profiles.
//    auto profile_events = worker.profiler.GetProfiles();
//  }
//
//
class BufferedProfiler : public tflite::Profiler {
 public:
  BufferedProfiler(uint32_t max_num_initial_entries,
                   bool allow_dynamic_buffer_increase)
      : buffer_(max_num_initial_entries, false /*enabled*/,
                allow_dynamic_buffer_increase),
        supported_event_types_(~static_cast<uint64_t>(
            EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_0(mht_0_v, 252, "", "./tensorflow/lite/profiling/buffered_profiler.h", "BufferedProfiler");
}

  explicit BufferedProfiler(uint32_t max_num_entries)
      : BufferedProfiler(max_num_entries,
                         false /*allow_dynamic_buffer_increase*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_1(mht_1_v, 259, "", "./tensorflow/lite/profiling/buffered_profiler.h", "BufferedProfiler");
}

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_2(mht_2_v, 267, "", "./tensorflow/lite/profiling/buffered_profiler.h", "BeginEvent");

    if (!ShouldAddEvent(event_type)) return kInvalidEventHandle;
    return buffer_.BeginEvent(tag, event_type, event_metadata1,
                              event_metadata2);
  }

  void EndEvent(uint32_t event_handle) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_3(mht_3_v, 276, "", "./tensorflow/lite/profiling/buffered_profiler.h", "EndEvent");

    buffer_.EndEvent(event_handle);
  }

  void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                int64_t event_metadata2) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_4(mht_4_v, 284, "", "./tensorflow/lite/profiling/buffered_profiler.h", "EndEvent");

    buffer_.EndEvent(event_handle, &event_metadata1, &event_metadata2);
  }

  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_5(mht_5_v, 294, "", "./tensorflow/lite/profiling/buffered_profiler.h", "AddEvent");

    if (!ShouldAddEvent(event_type)) return;
    buffer_.AddEvent(tag, event_type, start, end, event_metadata1,
                     event_metadata2);
  }

  void StartProfiling() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_6(mht_6_v, 303, "", "./tensorflow/lite/profiling/buffered_profiler.h", "StartProfiling");
 buffer_.SetEnabled(true); }
  void StopProfiling() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_7(mht_7_v, 307, "", "./tensorflow/lite/profiling/buffered_profiler.h", "StopProfiling");
 buffer_.SetEnabled(false); }
  void Reset() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_8(mht_8_v, 311, "", "./tensorflow/lite/profiling/buffered_profiler.h", "Reset");
 buffer_.Reset(); }
  std::vector<const ProfileEvent*> GetProfileEvents() {
    std::vector<const ProfileEvent*> profile_events;
    profile_events.reserve(buffer_.Size());
    for (size_t i = 0; i < buffer_.Size(); i++) {
      profile_events.push_back(buffer_.At(i));
    }
    return profile_events;
  }

 protected:
  bool ShouldAddEvent(EventType event_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_9(mht_9_v, 325, "", "./tensorflow/lite/profiling/buffered_profiler.h", "ShouldAddEvent");

    return (static_cast<uint64_t>(event_type) & supported_event_types_) != 0;
  }

 private:
  ProfileBuffer* GetProfileBuffer() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSbuffered_profilerDTh mht_10(mht_10_v, 333, "", "./tensorflow/lite/profiling/buffered_profiler.h", "GetProfileBuffer");
 return &buffer_; }
  ProfileBuffer buffer_;
  const uint64_t supported_event_types_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_
