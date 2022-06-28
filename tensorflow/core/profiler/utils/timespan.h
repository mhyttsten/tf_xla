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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TIMESPAN_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TIMESPAN_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh() {
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
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

// A Timespan is the time extent of an event: a pair of (begin, duration).
// Events may have duration 0 ("instant events") but duration can't be negative.
class Timespan {
 public:
  static Timespan FromEndPoints(uint64 begin_ps, uint64 end_ps) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/utils/timespan.h", "FromEndPoints");

    DCHECK_LE(begin_ps, end_ps);
    return Timespan(begin_ps, end_ps - begin_ps);
  }

  explicit Timespan(uint64 begin_ps = 0, uint64 duration_ps = 0)
      : begin_ps_(begin_ps), duration_ps_(duration_ps) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_1(mht_1_v, 212, "", "./tensorflow/core/profiler/utils/timespan.h", "Timespan");
}

  uint64 begin_ps() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_2(mht_2_v, 217, "", "./tensorflow/core/profiler/utils/timespan.h", "begin_ps");
 return begin_ps_; }
  uint64 middle_ps() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_3(mht_3_v, 221, "", "./tensorflow/core/profiler/utils/timespan.h", "middle_ps");
 return begin_ps_ + duration_ps_ / 2; }
  uint64 end_ps() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_4(mht_4_v, 225, "", "./tensorflow/core/profiler/utils/timespan.h", "end_ps");
 return begin_ps_ + duration_ps_; }
  uint64 duration_ps() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_5(mht_5_v, 229, "", "./tensorflow/core/profiler/utils/timespan.h", "duration_ps");
 return duration_ps_; }

  // Returns true if the Timespan represents an instant in time (duration 0).
  bool Instant() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_6(mht_6_v, 235, "", "./tensorflow/core/profiler/utils/timespan.h", "Instant");
 return duration_ps() == 0; }

  // Returns true if this is an empty timespan.
  bool Empty() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_7(mht_7_v, 241, "", "./tensorflow/core/profiler/utils/timespan.h", "Empty");
 return begin_ps() == 0 && duration_ps() == 0; }

  // Note for Overlaps() and Includes(Timespan& other) below:
  //   We have a design choice whether the end-point comparison should be
  //   inclusive or exclusive. We decide to go for inclusive. The implication
  //   is that an instant timespan could belong to two consecutive intervals
  //   (e.g., Timespan(12, 0) will be included in both Timespan(11, 1) and
  //   Timespan(12, 1)). We think this is okay because the common scenario
  //   would be that we search for the interval that includes a point
  //   in time from left to right, and return the first interval found.

  // Returns true if the Timespan overlaps with other.
  bool Overlaps(const Timespan& other) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_8(mht_8_v, 256, "", "./tensorflow/core/profiler/utils/timespan.h", "Overlaps");

    return begin_ps() <= other.end_ps() && other.begin_ps() <= end_ps();
  }

  // Returns true if this Timespan includes the other.
  bool Includes(const Timespan& other) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_9(mht_9_v, 264, "", "./tensorflow/core/profiler/utils/timespan.h", "Includes");

    return begin_ps() <= other.begin_ps() && other.end_ps() <= end_ps();
  }

  // Returns true if time_ps is within this Timespan.
  bool Includes(uint64 time_ps) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_10(mht_10_v, 272, "", "./tensorflow/core/profiler/utils/timespan.h", "Includes");
 return Includes(Timespan(time_ps)); }

  // Returns the duration in ps that this Timespan overlaps with the other.
  uint64 OverlappedDurationPs(const Timespan& other) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_11(mht_11_v, 278, "", "./tensorflow/core/profiler/utils/timespan.h", "OverlappedDurationPs");

    if (!Overlaps(other)) return 0;
    return std::min(end_ps(), other.end_ps()) -
           std::max(begin_ps(), other.begin_ps());
  }

  // Expands the timespan to include other.
  void ExpandToInclude(const Timespan& other) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_12(mht_12_v, 288, "", "./tensorflow/core/profiler/utils/timespan.h", "ExpandToInclude");

    *this = FromEndPoints(std::min(begin_ps(), other.begin_ps()),
                          std::max(end_ps(), other.end_ps()));
  }

  // Compares timespans by their begin time (ascending), duration (descending)
  // so nested spans are sorted from outer to innermost.
  bool operator<(const Timespan& other) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_13(mht_13_v, 298, "", "./tensorflow/core/profiler/utils/timespan.h", "operator<");

    if (begin_ps_ < other.begin_ps_) return true;
    if (begin_ps_ > other.begin_ps_) return false;
    return duration_ps_ > other.duration_ps_;
  }

  // Returns true if this timespan is equal to the given timespan.
  bool operator==(const Timespan& other) const {
    return begin_ps_ == other.begin_ps_ && duration_ps_ == other.duration_ps_;
  }

  // Returns a string that shows the begin and end times.
  std::string DebugString() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_14(mht_14_v, 313, "", "./tensorflow/core/profiler/utils/timespan.h", "DebugString");

    return absl::StrCat("[", begin_ps(), ", ", end_ps(), "]");
  }

  // Compares timespans by their duration_ps (ascending), begin time
  // (ascending).
  static bool ByDuration(const Timespan& a, const Timespan& b) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_15(mht_15_v, 322, "", "./tensorflow/core/profiler/utils/timespan.h", "ByDuration");

    if (a.duration_ps_ < b.duration_ps_) return true;
    if (a.duration_ps_ > b.duration_ps_) return false;
    return a.begin_ps_ < b.begin_ps_;
  }

 private:
  uint64 begin_ps_;
  uint64 duration_ps_;  // 0 for an instant event.
};

// Creates a Timespan from endpoints in picoseconds.
inline Timespan PicoSpan(uint64 start_ps, uint64 end_ps) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_16(mht_16_v, 337, "", "./tensorflow/core/profiler/utils/timespan.h", "PicoSpan");

  return Timespan::FromEndPoints(start_ps, end_ps);
}

// Creates a Timespan from endpoints in milliseconds.
inline Timespan MilliSpan(double start_ms, double end_ms) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStimespanDTh mht_17(mht_17_v, 345, "", "./tensorflow/core/profiler/utils/timespan.h", "MilliSpan");

  return PicoSpan(MilliToPico(start_ms), MilliToPico(end_ms));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TIMESPAN_H_
