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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc() {
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

#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"

#include <functional>
#include <list>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace {

absl::Mutex mu(absl::kConstInit);
absl::CondVar* ready;
absl::once_flag init_flag;
std::list<SlowOperationAlarm*>* outstanding_alarms ABSL_PT_GUARDED_BY(mu) =
    nullptr;

}  // namespace

void SlowOperationAlarm::AlarmLoop() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::AlarmLoop");

  while (true) {
    absl::MutexLock lock(&mu);

    // Fire any alarms which are ready.
    absl::Time now = absl::Now();
    for (auto it = outstanding_alarms->begin();
         it != outstanding_alarms->end();) {
      auto next = std::next(it);
      auto* alarm = *it;
      // Fire the alarm if applicable.
      if (alarm->deadline() <= now) {
        outstanding_alarms->erase(it);
        int64_t count =
            alarm->counter() == nullptr ? 0 : alarm->counter()->fetch_add(1);
        // If the alarm has a counter, only fire if the count is a power of 2.
        if (count == 0 || (count & (count - 1)) == 0) {
          alarm->fired_.store(true);
          // We fire alarms with LOG(ERROR) because otherwise it might not show
          // up without --logtostderr.
          LOG(ERROR) << alarm->msg();
        }
      }
      it = next;
    }

    if (outstanding_alarms->empty()) {
      ready->Wait(&mu);
      continue;
    }

    SlowOperationAlarm* next_alarm = *absl::c_min_element(
        *outstanding_alarms,
        [](const SlowOperationAlarm* a, const SlowOperationAlarm* b) {
          return a->deadline() < b->deadline();
        });
    ready->WaitWithDeadline(&mu, next_alarm->deadline());
  }
}

void SlowOperationAlarm::ScheduleAlarm(SlowOperationAlarm* alarm) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_1(mht_1_v, 253, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::ScheduleAlarm");

  absl::call_once(init_flag, [] {
    ready = new absl::CondVar();
    outstanding_alarms = new std::list<SlowOperationAlarm*>();
    (void)tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "SlowOperationAlarm", [] { AlarmLoop(); });
  });

  absl::MutexLock lock(&mu);
  outstanding_alarms->push_back(alarm);
  ready->Signal();
}

void SlowOperationAlarm::UnscheduleAlarm(const SlowOperationAlarm* alarm) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_2(mht_2_v, 269, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::UnscheduleAlarm");

  absl::MutexLock lock(&mu);
  CHECK(outstanding_alarms != nullptr);
  auto it = absl::c_find(*outstanding_alarms, alarm);
  if (it != outstanding_alarms->end()) {
    outstanding_alarms->erase(it);
  }
}
SlowOperationAlarm::SlowOperationAlarm(
    absl::Duration timeout, std::string msg,
    std::atomic<int64_t>* counter /*=nullptr*/)
    : SlowOperationAlarm(
          timeout,
          // TODO(b/157309856): Once we have C++17, capture msg "by move".
          [msg] { return msg; },  //
          counter) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("msg: \"" + msg + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::SlowOperationAlarm");
}

SlowOperationAlarm::SlowOperationAlarm(
    absl::Duration timeout, std::function<std::string()> msg_fn,
    std::atomic<int64_t>* counter /*=nullptr*/)
    : deadline_(absl::Now() + timeout),
      msg_fn_(std::move(msg_fn)),
      counter_(counter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_4(mht_4_v, 298, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::SlowOperationAlarm");

  ScheduleAlarm(this);
}

SlowOperationAlarm::~SlowOperationAlarm() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSslow_operation_alarmDTcc mht_5(mht_5_v, 305, "", "./tensorflow/compiler/xla/service/slow_operation_alarm.cc", "SlowOperationAlarm::~SlowOperationAlarm");
 UnscheduleAlarm(this); }

std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    absl::string_view msg) {
  // Pass a counter to these alarms so they only log once every power-of-two
  // occurrences.
  static auto* counter = new std::atomic<int64_t>(0);

  const char* separator = "\n********************************";

  std::string msg_suffix;
  if (!msg.empty()) {
    msg_suffix = absl::StrCat("\n", msg);
  }

#if NDEBUG
  return absl::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Minutes(2)),
      absl::StrCat(
          separator,
          "\nVery slow compile?  If you want to file a bug, run with envvar "
          "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.",
          msg_suffix, separator),
      counter);
#else
  return absl::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Seconds(10)),
      absl::StrCat(
          separator,
          "\nSlow compile?  XLA was built without compiler optimizations, "
          "which can be slow.  Try rebuilding with -c opt.",
          msg_suffix, separator),
      counter);
#endif
}

}  // namespace xla
