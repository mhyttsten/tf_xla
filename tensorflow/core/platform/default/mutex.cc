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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc() {
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

#include "tensorflow/core/platform/mutex.h"

#include <time.h>

#include "nsync_cv.h"       // NOLINT
#include "nsync_mu.h"       // NOLINT
#include "nsync_mu_wait.h"  // NOLINT
#include "nsync_time.h"     // NOLINT

namespace tensorflow {

// Check that the MuData struct used to reserve space for the mutex
// in tensorflow::mutex is big enough.
static_assert(sizeof(nsync::nsync_mu) <= sizeof(internal::MuData),
              "tensorflow::internal::MuData needs to be bigger");

// Cast a pointer to internal::MuData to a pointer to the mutex
// representation.  This is done so that the header files for nsync_mu do not
// need to be included in every file that uses tensorflow's mutex.
static inline nsync::nsync_mu *mu_cast(internal::MuData *mu) {
  return reinterpret_cast<nsync::nsync_mu *>(mu);
}

mutex::mutex() { nsync::nsync_mu_init(mu_cast(&mu_)); }

mutex::mutex(LinkerInitialized x) {}

void mutex::lock() { nsync::nsync_mu_lock(mu_cast(&mu_)); }

bool mutex::try_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/platform/default/mutex.cc", "mutex::try_lock");
 return nsync::nsync_mu_trylock(mu_cast(&mu_)) != 0; };

void mutex::unlock() { nsync::nsync_mu_unlock(mu_cast(&mu_)); }

void mutex::lock_shared() { nsync::nsync_mu_rlock(mu_cast(&mu_)); }

bool mutex::try_lock_shared() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/platform/default/mutex.cc", "mutex::try_lock_shared");

  return nsync::nsync_mu_rtrylock(mu_cast(&mu_)) != 0;
};

void mutex::unlock_shared() { nsync::nsync_mu_runlock(mu_cast(&mu_)); }

// A callback suitable for nsync_mu_wait() that calls Condition::Eval().
static int EvaluateCondition(const void *vcond) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/platform/default/mutex.cc", "EvaluateCondition");

  return static_cast<int>(static_cast<const Condition *>(vcond)->Eval());
}

void mutex::Await(const Condition &cond) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/platform/default/mutex.cc", "mutex::Await");

  nsync::nsync_mu_wait(mu_cast(&mu_), &EvaluateCondition, &cond, nullptr);
}

bool mutex::AwaitWithDeadline(const Condition &cond, uint64 abs_deadline_ns) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/platform/default/mutex.cc", "mutex::AwaitWithDeadline");

  time_t seconds = abs_deadline_ns / (1000 * 1000 * 1000);
  nsync::nsync_time abs_time = nsync::nsync_time_s_ns(
      seconds, abs_deadline_ns - seconds * (1000 * 1000 * 1000));
  return nsync::nsync_mu_wait_with_deadline(mu_cast(&mu_), &EvaluateCondition,
                                            &cond, nullptr, abs_time,
                                            nullptr) == 0;
}

// Check that the CVData struct used to reserve space for the
// condition variable in tensorflow::condition_variable is big enough.
static_assert(sizeof(nsync::nsync_cv) <= sizeof(internal::CVData),
              "tensorflow::internal::CVData needs to be bigger");

// Cast a pointer to internal::CVData to a pointer to the condition
// variable representation.  This is done so that the header files for nsync_cv
// do not need to be included in every file that uses tensorflow's
// condition_variable.
static inline nsync::nsync_cv *cv_cast(internal::CVData *cv) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/platform/default/mutex.cc", "cv_cast");

  return reinterpret_cast<nsync::nsync_cv *>(cv);
}

condition_variable::condition_variable() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_6(mht_6_v, 275, "", "./tensorflow/core/platform/default/mutex.cc", "condition_variable::condition_variable");

  nsync::nsync_cv_init(cv_cast(&cv_));
}

void condition_variable::wait(mutex_lock &lock) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_7(mht_7_v, 282, "", "./tensorflow/core/platform/default/mutex.cc", "condition_variable::wait");

  nsync::nsync_cv_wait(cv_cast(&cv_), mu_cast(&lock.mutex()->mu_));
}

void condition_variable::notify_one() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_8(mht_8_v, 289, "", "./tensorflow/core/platform/default/mutex.cc", "condition_variable::notify_one");
 nsync::nsync_cv_signal(cv_cast(&cv_)); }

void condition_variable::notify_all() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_9(mht_9_v, 294, "", "./tensorflow/core/platform/default/mutex.cc", "condition_variable::notify_all");

  nsync::nsync_cv_broadcast(cv_cast(&cv_));
}

namespace internal {
std::cv_status wait_until_system_clock(
    CVData *cv_data, MuData *mu_data,
    const std::chrono::system_clock::time_point timeout_time) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSmutexDTcc mht_10(mht_10_v, 304, "", "./tensorflow/core/platform/default/mutex.cc", "wait_until_system_clock");

  int r = nsync::nsync_cv_wait_with_deadline(cv_cast(cv_data), mu_cast(mu_data),
                                             timeout_time, nullptr);
  return r ? std::cv_status::timeout : std::cv_status::no_timeout;
}
}  // namespace internal

}  // namespace tensorflow
