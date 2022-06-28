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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"

#include <fcntl.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <fstream>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/tools/logging.h"

#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_entrypoint.h"
#endif  // __ANDROID__

namespace tflite {
namespace acceleration {
namespace {
#ifdef __ANDROID__
void* LoadEntryPointModule(const std::string& module_path) {
  void* module =
      dlopen(module_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (module == nullptr) {
    TFLITE_LOG(FATAL) << dlerror();
  }
  return module;
}
#endif  // __ANDROID__

std::string JoinPath(const std::string& path1, const std::string& path2) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path1: \"" + path1 + "\"");
   mht_0_v.push_back("path2: \"" + path2 + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc mht_0(mht_0_v, 221, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.cc", "JoinPath");

  if (path1.empty()) return path2;
  if (path2.empty()) return path1;
  if (path1.back() == '/') {
    if (path2.front() == '/') return path1 + path2.substr(1);
  } else {
    if (path2.front() != '/') return path1 + "/" + path2;
  }
  return path1 + path2;
}
}  // namespace

MiniBenchmarkTestHelper::MiniBenchmarkTestHelper()
    : should_perform_test_(true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc mht_1(mht_1_v, 237, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.cc", "MiniBenchmarkTestHelper::MiniBenchmarkTestHelper");

#ifdef __ANDROID__
  AndroidInfo android_info;
  const auto status = RequestAndroidInfo(&android_info);
  if (!status.ok() || android_info.is_emulator) {
    should_perform_test_ = false;
    return;
  }

  DumpToTempFile("librunner_main.so", g_tflite_acceleration_embedded_runner,
                 g_tflite_acceleration_embedded_runner_len);

  std::string validator_runner_so_path = DumpToTempFile(
      "libvalidator_runner_entrypoint.so",
      g_tflite_acceleration_embedded_validator_runner_entrypoint,
      g_tflite_acceleration_embedded_validator_runner_entrypoint_len);
  // Load this library here because it contains the validation entry point
  // "Java_org_tensorflow_lite_acceleration_validation_entrypoint" that is then
  // found using dlsym (using RTLD_DEFAULT hence not needing the handle) in the
  // mini-benchmark code.
  LoadEntryPointModule(validator_runner_so_path);
#endif  // __ANDROID__
}

std::string MiniBenchmarkTestHelper::DumpToTempFile(const std::string& filename,
                                                    const unsigned char* data,
                                                    size_t length) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   mht_2_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_test_helperDTcc mht_2(mht_2_v, 268, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.cc", "MiniBenchmarkTestHelper::DumpToTempFile");

  std::string path = JoinPath(::testing::TempDir(), filename);
  (void)unlink(path.c_str());
  std::string contents(reinterpret_cast<const char*>(data), length);
  std::ofstream f(path, std::ios::binary);
  EXPECT_TRUE(f.is_open());
  f << contents;
  f.close();
  EXPECT_EQ(0, chmod(path.c_str(), 0500));
  return path;
}

}  // namespace acceleration
}  // namespace tflite
