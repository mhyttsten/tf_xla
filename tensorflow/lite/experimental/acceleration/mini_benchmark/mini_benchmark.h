/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTh() {
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


#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {
// Instances are thread-compatible, access from multiple threads must be guarded
// by a mutex.
//
// Caution: The mini-benchmark runs silently in-process on non-Android, rather
// than in a separate process.
class MiniBenchmark {
 public:
  // Get best acceleration based on tests done so far. If no successful tests
  // are found, the best settings are on CPU or if the settings do not contain
  // configurations to test or not all relevant fields are present, the returned
  // ComputeSettingsT will be an object created by the default constructor.
  // Note: if we have successful mini-benchmark run events, the best
  // acceleration configuration will be persisted on the local storage as a new
  // mini-benchmark event.
  virtual ComputeSettingsT GetBestAcceleration() = 0;

  // Trigger the running of tests in the background in a separate thread on
  // Linux, but a separate process on Android. If triggering fails, errors are
  // stored internally.
  //
  // Does nothing if the settings do not contain configurations to test or not
  // all relevant fields are present.
  virtual void TriggerMiniBenchmark() = 0;

  virtual void SetEventTimeoutForTesting(int64_t timeout_us) = 0;

  // Mark mini-benchmark events that have not yet been marked as to be logged,
  // and return these events. This can include errors in triggering the
  // mini-benchmark.
  virtual std::vector<MiniBenchmarkEventT> MarkAndGetEventsToLog() = 0;

  // Get the number of remaining tests that still need to be completed.
  // Note that this method should be only called after calling
  // TriggerMiniBenchmark or GetBestAcceleration where additional
  // mini-benchmark-related setup could be initialized. In short, -1 is returned
  // if the overall mini-benchmark-related setup isn't properly initialized.
  virtual int NumRemainingAccelerationTests() = 0;

  MiniBenchmark() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTh mht_0(mht_0_v, 236, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.h", "MiniBenchmark");
}
  virtual ~MiniBenchmark() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTh mht_1(mht_1_v, 240, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.h", "~MiniBenchmark");
}

  MiniBenchmark(MiniBenchmark&) = delete;
  MiniBenchmark& operator=(const MiniBenchmark&) = delete;
  MiniBenchmark(MiniBenchmark&&) = delete;
  MiniBenchmark& operator=(const MiniBenchmark&&) = delete;
};

// Instantiate a mini-benchmark. This will return a no-op implementation unless
// the :mini_benchmark_implementation target has been linked in.
std::unique_ptr<MiniBenchmark> CreateMiniBenchmark(
    const MinibenchmarkSettings& settings, const std::string& model_namespace,
    const std::string& model_id);

// A simple registry that allows different mini-benchmark implementations to be
// created by name.
//
// Limitations:
// - Doesn't allow deregistration.
// - Doesn't check for duplication registration.
//
class MinibenchmarkImplementationRegistry {
 public:
  using CreatorFunction = std::function<std::unique_ptr<MiniBenchmark>(
      const MinibenchmarkSettings& /*settings*/,
      const std::string& /*model_namespace*/, const std::string& /*model_id*/)>;

  // Returns a MiniBenchmark registered with `name` or nullptr if no matching
  // mini-benchmark implementation found.
  static std::unique_ptr<MiniBenchmark> CreateByName(
      const std::string& name, const MinibenchmarkSettings& settings,
      const std::string& model_namespace, const std::string& model_id);

  // Struct to be statically allocated for registration.
  struct Register {
    Register(const std::string& name, CreatorFunction creator_function);
  };

 private:
  void RegisterImpl(const std::string& name, CreatorFunction creator_function);
  std::unique_ptr<MiniBenchmark> CreateImpl(
      const std::string& name, const MinibenchmarkSettings& settings,
      const std::string& model_namespace, const std::string& model_id);
  static MinibenchmarkImplementationRegistry* GetSingleton();

  absl::Mutex mutex_;
  std::unordered_map<std::string, CreatorFunction> factories_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace acceleration
}  // namespace tflite

#define TFLITE_REGISTER_MINI_BENCMARK_FACTORY_FUNCTION(name, f) \
  static auto* g_tflite_mini_benchmark_##name##_ =              \
      new MinibenchmarkImplementationRegistry::Register(#name, f);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_
