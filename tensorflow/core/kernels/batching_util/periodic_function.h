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

// PeriodicFunction will periodically call the given function with a specified
// period in a background thread.  After Start() returns, the thread is
// guaranteed to have started. The destruction of the class causes the
// background thread to be destroyed as well.  Start() should not be called more
// than once.
//
// PeriodicFunction runs the function as soon as any previous run both is
// complete and was started more than "interval_micros" earlier.  Thus, runs are
// both serialized, and normally have a period of "interval_micros" if no run
// exceeds the time.
//
// Note that, if the function takes longer than two interval_micross to finish,
// then PeriodicFunction will "skip" at least one call to the function.  For
// instance, if the period is 50ms and the function starts runs at time 0 for
// 150ms, then the function will immediately start executing again at time 150,
// but there will be no function runs corresponding to times 50 or 100.  This is
// especially important to remember when using an environment with a simulated
// clock: advancing simulated time atomically over N interval_micross will not
// cause the function to be called N times.
//
// This object is thread-safe.
//
// Example:
//
//   class Foo {
//    public:
//     Foo() : periodic_function_([this]() { Bar(); },
//                               1000 /* 1000us == 1ms*/) {
//     }
//
//    private:
//     void Bar() { ... }
//
//     PeriodicFunction periodic_function_;
//   };

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_PERIODIC_FUNCTION_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_PERIODIC_FUNCTION_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_functionDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_functionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_functionDTh() {
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


#include "tensorflow/core/kernels/batching_util/periodic_function.h"

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

namespace internal {
class PeriodicFunctionTestAccess;
}

class PeriodicFunction {
 public:
  // Provides the ability to customize several aspects of the PeriodicFunction.
  // Passed to constructor of PeriodicFunction.
  struct Options {
    Options() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbatching_utilPSperiodic_functionDTh mht_0(mht_0_v, 247, "", "./tensorflow/core/kernels/batching_util/periodic_function.h", "Options");
}

    // Any standard thread options, such as stack size, should
    // be passed via "thread_options".
    ThreadOptions thread_options;

    // Specifies the thread name prefix (see the description in class
    // Thread).
    string thread_name_prefix = "periodic_function";

    // The environment to use. Does not take ownership, but must remain alive
    // for as long as the PeriodicFunction exists.
    Env* env = Env::Default();

    // Specifies the length of sleep before the first invocation of the
    // function.
    // This can be used for adding a random jitter to avoid synchronous behavior
    // across multiple periodic functions.
    int64_t startup_delay_micros = 0;
  };

  // Also starts the background thread which will be calling the function.
  PeriodicFunction(const std::function<void()>& function,
                   int64_t interval_micros, const Options& options = Options());

  ~PeriodicFunction();

 private:
  friend class internal::PeriodicFunctionTestAccess;

  // Notifies the background thread to stop.
  void NotifyStop();

  // (Blocking.) Loops forever calling "function_" every "interval_micros_".
  void RunLoop(int64_t start);

  const std::function<void()> function_;  // Actual client function
  const int64_t interval_micros_;         // Interval between calls.
  const Options options_;

  // Used to notify the thread to stop.
  Notification stop_thread_;

  // Thread for running "function_"
  std::unique_ptr<Thread> thread_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(PeriodicFunction);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_PERIODIC_FUNCTION_H_
