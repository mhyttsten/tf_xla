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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh() {
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


#include <fcntl.h>
#ifndef _WIN32
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {

constexpr const char* TfLiteValidationFunctionName() {
  return "Java_org_tensorflow_lite_acceleration_validation_entrypoint";
}

// Class that runs mini-benchmark validation in a separate process and gives
// access to the results.
//
// It is safe to construct more than one instance of the ValidatorRunner in one
// or more processes. File locks are used to ensure the storage is mutated
// safely and that we run at most one validation at a time for a given
// data_directory_path.
//
// A single instance of ValidatorRunner is thread-compatible (access from
// multiple threads must be guarded with a mutex).
class ValidatorRunner {
 public:
  static constexpr int64_t kDefaultEventTimeoutUs = 30 * 1000 * 1000;

  // Construct ValidatorRunner for a model and a file for storing results in.
  // The 'storage_path' must be specific for the model.
  // 'data_directory_path' must be suitable for extracting an executable file
  // to.
  // The nnapi_sl pointer can be used to configure the runner to use
  // the NNAPI implementation coming from the Support Library instead of
  // the NNAPI platform drivers.
  // If nnapi_sl is not null we expect the functions referenced by the structure
  // lifetime to be enclosing the one of the mini-benchmark. In particular
  // we expect that if the NnApiSupportLibrary was loaded by a shared library,
  // dlclose is called only after all this mini-benchmark object has been
  // deleted.
  ValidatorRunner(const std::string& model_path,
                  const std::string& storage_path,
                  const std::string& data_directory_path,
                  const NnApiSLDriverImplFL5* nnapi_sl = nullptr,
                  const std::string validation_function_name =
                      TfLiteValidationFunctionName(),
                  ErrorReporter* error_reporter = DefaultErrorReporter());
  ValidatorRunner(int model_fd, size_t model_offset, size_t model_size,
                  const std::string& storage_path,
                  const std::string& data_directory_path,
                  const NnApiSLDriverImplFL5* nnapi_sl = nullptr,
                  const std::string validation_function_name =
                      TfLiteValidationFunctionName(),
                  ErrorReporter* error_reporter = DefaultErrorReporter());
  MinibenchmarkStatus Init();

  // The following methods invalidate previously returned pointers.

  // Run validation for those settings in 'for_settings' where validation has
  // not yet been run. Incomplete validation may be retried a small number of
  // times (e.g., 2).
  // Returns number of runs triggered (this may include runs triggered through a
  // different instance, and is meant for debugging).
  int TriggerMissingValidation(std::vector<const TFLiteSettings*> for_settings);
  // Get results for successfully completed validation runs. The caller can then
  // pick the best configuration based on timings.
  std::vector<const BenchmarkEvent*> GetSuccessfulResults();
  // Get results for completed validation runs regardless whether it is
  // successful or not.
  int GetNumCompletedResults();
  // Get all relevant results for telemetry. Will contain:
  // - Start events if an incomplete test is found. Tests are considered
  // incomplete, if they started more than timeout_us ago and do not have
  // results/errors.
  // - Error events where the test ended with an error
  // - End events where the test was completed (even if results were incorrect).
  // The returned events will be marked as logged and not returned again on
  // subsequent calls.
  std::vector<const BenchmarkEvent*> GetAndFlushEventsToLog(
      int64_t timeout_us = kDefaultEventTimeoutUs);

 private:
  std::string model_path_;
  int model_fd_ = -1;
  size_t model_offset_, model_size_;
  std::string storage_path_;
  std::string data_directory_path_;
  FlatbufferStorage<BenchmarkEvent> storage_;
  std::string validation_function_name_;
  ErrorReporter* error_reporter_;
  bool triggered_ = false;
  std::string nnapi_sl_path_;
  const NnApiSLDriverImplFL5* nnapi_sl_;
};

}  // namespace acceleration
}  // namespace tflite

class FileLock {
 public:
  explicit FileLock(const std::string& path) : path_(path) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh mht_0(mht_0_v, 300, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h", "FileLock");
}
  bool TryLock() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh mht_1(mht_1_v, 304, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h", "TryLock");

#ifndef _WIN32  // Validator runner not supported on Windows.
    // O_CLOEXEC is needed for correctness, as another thread may call
    // popen() and the callee inherit the lock if it's not O_CLOEXEC.
    fd_ = open(path_.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
    if (fd_ < 0) {
      return false;
    }
    if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
      return true;
    }
#endif  // !_WIN32
    return false;
  }
  ~FileLock() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runnerDTh mht_2(mht_2_v, 321, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h", "~FileLock");

#ifndef _WIN32  // Validator runner not supported on Windows.
    if (fd_ >= 0) {
      close(fd_);
    }
#endif  // !_WIN32
  }

 private:
  std::string path_;
  int fd_ = -1;
};

extern "C" {
int Java_org_tensorflow_lite_acceleration_validation_entrypoint(int argc,
                                                                char** argv);
}  // extern "C"

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_VALIDATOR_RUNNER_H_
