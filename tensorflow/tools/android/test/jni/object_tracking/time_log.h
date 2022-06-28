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

// Utility functions for performance profiling.

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh() {
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


#include <stdint.h>

#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

#ifdef LOG_TIME

// Blend constant for running average.
#define ALPHA 0.98f
#define NUM_LOGS 100

struct LogEntry {
  const char* id;
  int64_t time_stamp;
};

struct AverageEntry {
  const char* id;
  float average_duration;
};

// Storage for keeping track of this frame's values.
extern int num_time_logs;
extern LogEntry time_logs[NUM_LOGS];

// Storage for keeping track of average values (each entry may not be printed
// out each frame).
extern AverageEntry avg_entries[NUM_LOGS];
extern int num_avg_entries;

// Call this at the start of a logging phase.
inline static void ResetTimeLog() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_0(mht_0_v, 221, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "ResetTimeLog");

  num_time_logs = 0;
}


// Log a message to be printed out when printTimeLog is called, along with the
// amount of time in ms that has passed since the last call to this function.
inline static void TimeLog(const char* const str) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_1(mht_1_v, 231, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "TimeLog");

  LOGV("%s", str);
  if (num_time_logs >= NUM_LOGS) {
    LOGE("Out of log entries!");
    return;
  }

  time_logs[num_time_logs].id = str;
  time_logs[num_time_logs].time_stamp = CurrentThreadTimeNanos();
  ++num_time_logs;
}


inline static float Blend(float old_val, float new_val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_2(mht_2_v, 247, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "Blend");

  return ALPHA * old_val + (1.0f - ALPHA) * new_val;
}


inline static float UpdateAverage(const char* str, const float new_val) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_3(mht_3_v, 256, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "UpdateAverage");

  for (int entry_num = 0; entry_num < num_avg_entries; ++entry_num) {
    AverageEntry* const entry = avg_entries + entry_num;
    if (str == entry->id) {
      entry->average_duration = Blend(entry->average_duration, new_val);
      return entry->average_duration;
    }
  }

  if (num_avg_entries >= NUM_LOGS) {
    LOGE("Too many log entries!");
  }

  // If it wasn't there already, add it.
  avg_entries[num_avg_entries].id = str;
  avg_entries[num_avg_entries].average_duration = new_val;
  ++num_avg_entries;

  return new_val;
}


// Prints out all the timeLog statements in chronological order with the
// interval that passed between subsequent statements.  The total time between
// the first and last statements is printed last.
inline static void PrintTimeLog() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_4(mht_4_v, 284, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "PrintTimeLog");

  LogEntry* last_time = time_logs;

  float average_running_total = 0.0f;

  for (int i = 0; i < num_time_logs; ++i) {
    LogEntry* const this_time = time_logs + i;

    const float curr_time =
        (this_time->time_stamp - last_time->time_stamp) / 1000000.0f;

    const float avg_time = UpdateAverage(this_time->id, curr_time);
    average_running_total += avg_time;

    LOGD("%32s:    %6.3fms    %6.4fms", this_time->id, curr_time, avg_time);
    last_time = this_time;
  }

  const float total_time =
      (last_time->time_stamp - time_logs->time_stamp) / 1000000.0f;

  LOGD("TOTAL TIME:                          %6.3fms    %6.4fms\n",
       total_time, average_running_total);
  LOGD(" ");
}
#else
inline static void ResetTimeLog() {}

inline static void TimeLog(const char* const str) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_5(mht_5_v, 315, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "TimeLog");

  LOGV("%s", str);
}

inline static void PrintTimeLog() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStime_logDTh mht_6(mht_6_v, 322, "", "./tensorflow/tools/android/test/jni/object_tracking/time_log.h", "PrintTimeLog");
}
#endif

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TIME_LOG_H_
