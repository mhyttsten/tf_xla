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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh() {
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
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// This RAII object asynchronously prints a warning if it's alive for more than
// a certain amount of time.
class SlowOperationAlarm {
 public:
  // If `counter` is not null, this alarm will throttle itself to logging
  // once-every-power-of-two occurrences. The counter must outlive this object.
  SlowOperationAlarm(absl::Duration timeout, std::string msg,
                     std::atomic<int64_t>* counter = nullptr);
  SlowOperationAlarm(absl::Duration timeout,
                     std::function<std::string()> msg_fn,
                     std::atomic<int64_t>* counter = nullptr);
  ~SlowOperationAlarm();

  // Not copyable or movable, because the constructor stores a pointer to `this`
  // into a global variable.
  SlowOperationAlarm(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm(const SlowOperationAlarm&&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&) = delete;
  SlowOperationAlarm& operator=(const SlowOperationAlarm&&) = delete;

  absl::Time deadline() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.h", "deadline");
 return deadline_; }
  std::string msg() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh mht_1(mht_1_v, 225, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.h", "msg");
 return msg_fn_(); }
  std::atomic<int64_t>* counter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.h", "counter");
 return counter_; }
  void cancel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh mht_3(mht_3_v, 233, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.h", "cancel");
 UnscheduleAlarm(this); }
  // Has the alarm fired?  If appropriate, consider cancel()'ing first, to avoid
  // a race.
  bool fired() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTh mht_4(mht_4_v, 239, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.h", "fired");
 return fired_.load(); }

 private:
  static void AlarmLoop();
  static void ScheduleAlarm(SlowOperationAlarm* alarm);
  static void UnscheduleAlarm(const SlowOperationAlarm* alarm);

  absl::Time deadline_;
  std::function<std::string()> msg_fn_;
  std::atomic<bool> fired_{false};
  // counter_ may be null.  If it's not, this alarm prints something only once
  // every power of two occurrences.
  std::atomic<int64_t>* counter_;
};

// Returns an object which prints a warning about slow compilation after a
// certain amount of time.
//
// In debug builds, recommends building with -c opt.
//
// In opt builds, recommends filing a bug.
//
// This is throttled to once-every-power-of-two occurrences, globally.
ABSL_MUST_USE_RESULT std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    absl::string_view msg = "");

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SLOW_OPERATION_ALARM_H_
