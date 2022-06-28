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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"

#include <dlfcn.h>
#include <signal.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_unit_test_entry_points.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#endif  // __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

extern "C" {
int FunctionInBinary(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/runner_test.cc", "FunctionInBinary");
 return 2; }
}  // extern "C"

namespace tflite {
namespace acceleration {
namespace {

typedef int (*EntryPoint)(int, char**);
struct RunnerTest : ::testing::Test {
 protected:
  void* LoadEntryPointModule() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/runner_test.cc", "LoadEntryPointModule");

    void* module =
        dlopen(entry_point_file.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    EXPECT_TRUE(module) << dlerror();
    return module;
  }

  EntryPoint Load(const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/runner_test.cc", "Load");

#ifdef __ANDROID__
    void* module = LoadEntryPointModule();
    if (!module) {
      return nullptr;
    }
#else   // !__ANDROID__
    auto module = RTLD_DEFAULT;
#endif  // __ANDROID__
    void* symbol = dlsym(module, name);
    return reinterpret_cast<EntryPoint>(symbol);
  }

  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/runner_test.cc", "SetUp");

#ifdef __ANDROID__
    // We extract the test files here as that's the only way to get the right
    // architecture when building tests for multiple architectures.
    entry_point_file = MiniBenchmarkTestHelper::DumpToTempFile(
        "librunner_unit_test_entry_points.so",
        g_tflite_acceleration_embedded_runner_unit_test_entry_points,
        g_tflite_acceleration_embedded_runner_unit_test_entry_points_len);
    ASSERT_TRUE(!entry_point_file.empty());
#endif  // __ANDROID__
  }

  void Init(const char* symbol_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSrunner_testDTcc mht_4(mht_4_v, 260, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/runner_test.cc", "Init");

    EntryPoint function = Load(symbol_name);
    ASSERT_TRUE(function);
    runner = std::make_unique<ProcessRunner>(::testing::TempDir(), symbol_name,
                                             function);
    ASSERT_EQ(runner->Init(), kMinibenchmarkSuccess);
  }
  int exitcode = 0;
  int signal = 0;
  std::string output;
  std::unique_ptr<ProcessRunner> runner;
  std::string entry_point_file;
};

// These tests are only for Android. They are also disabled on 64-bit arm
// because the 64-bit arm emulator doesn't have a shell that works with popen().
// These tests are run on x86 emulators.
#if !defined(__aarch64__)
TEST_F(RunnerTest, LoadSymbol) {
  EntryPoint JustReturnZero = Load("JustReturnZero");
  ASSERT_TRUE(JustReturnZero);
#ifdef __ANDROID__
  Dl_info dl_info;
  int status = dladdr(reinterpret_cast<void*>(JustReturnZero), &dl_info);
  ASSERT_TRUE(status) << dlerror();
  ASSERT_TRUE(dl_info.dli_fname) << dlerror();

  void* this_module =
      dlopen(dl_info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  ASSERT_TRUE(this_module);
  void* symbol = dlsym(this_module, "JustReturnZero");
  EXPECT_TRUE(symbol);
#endif  // __ANDROID__
}

TEST_F(RunnerTest, JustReturnZero) {
  ASSERT_NO_FATAL_FAILURE(Init("JustReturnZero"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnOne) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnOne"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 1);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnSuccess) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnSuccess"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

#ifdef __ANDROID__
TEST_F(RunnerTest, SigKill) {
  ASSERT_NO_FATAL_FAILURE(Init("SigKill"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, SIGKILL);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, WriteOk) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteOk"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "ok\n");
}

TEST_F(RunnerTest, Write10kChars) {
  ASSERT_NO_FATAL_FAILURE(Init("Write10kChars"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output.size(), 10000);
}

TEST_F(RunnerTest, ArgsArePassed) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteArgs"));
  EXPECT_EQ(runner->Run({"foo", "bar", "baz"}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "foo\nbar\nbaz\n");
}
#endif  // __ANDROID__

TEST_F(RunnerTest, NullFunctionPointer) {
  ProcessRunner runner("foo", "bar", nullptr);
  EXPECT_EQ(runner.Init(), kMinibenchmarkPreconditionNotMet);
  EXPECT_EQ(runner.Run({}, &output, &exitcode, &signal),
            kMinibenchmarkPreconditionNotMet);
}

#ifdef __ANDROID__
TEST_F(RunnerTest, SymbolLookupFailed) {
  ProcessRunner runner(::testing::TempDir(), "FunctionInBinary",
                       FunctionInBinary);
  EXPECT_EQ(runner.Init(), kMinibenchmarkSuccess);
  EXPECT_EQ(runner.Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed)
      << output;
  EXPECT_EQ(exitcode, kMinibenchmarkRunnerMainSymbolLookupFailed) << output;
}
#endif  // __ANDROID__
#endif  // !__aarch64__

}  // namespace
}  // namespace acceleration
}  // namespace tflite
