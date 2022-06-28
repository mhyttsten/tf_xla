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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc() {
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
#include <sys/types.h>

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h"

// Note that these tests are not meant to be completely exhaustive, but to test
// error propagation.

namespace tflite {
namespace acceleration {

static int32_t big_core_affinity_result;
int32_t SetBigCoresAffinity() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "SetBigCoresAffinity");
 return big_core_affinity_result; }

namespace {

class ValidatorRunnerEntryPointTest : public ::testing::Test {
 protected:
  ValidatorRunnerEntryPointTest()
      : storage_path_(::testing::TempDir() + "/events.fb"),
        storage_(storage_path_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "ValidatorRunnerEntryPointTest");
}

  std::vector<const tflite::BenchmarkEvent*> GetEvents() {
    std::vector<const tflite::BenchmarkEvent*> result;

    storage_.Read();
    int storage_size = storage_.Count();
    if (storage_size == 0) {
      return result;
    }

    for (int i = 0; i < storage_size; i++) {
      const ::tflite::BenchmarkEvent* event = storage_.Get(i);
      result.push_back(event);
    }
    return result;
  }

  void ClearEvents() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "ClearEvents");
 (void)unlink(storage_path_.c_str()); }

  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "SetUp");

    ClearEvents();
    SetBigCoreAffinityReturns(0);
  }

  int CallEntryPoint(std::string cpu_affinity = "0") {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_4(mht_4_v, 249, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "CallEntryPoint");

    std::vector<std::string> args = {
        "test",
        "binary_name",
        "Java_org_tensorflow_lite_acceleration_validation_entrypoint",
        "model_path",
        storage_path_,
        "data_dir"};
    std::vector<std::vector<char>> mutable_args(args.size());
    std::vector<char*> argv(args.size());
    for (int i = 0; i < mutable_args.size(); i++) {
      mutable_args[i] = {args[i].data(), args[i].data() + args[i].size()};
      mutable_args[i].push_back('\0');
      argv[i] = mutable_args[i].data();
    }
    return Java_org_tensorflow_lite_acceleration_validation_entrypoint(
        argv.size(), argv.data());
  }

  void SetBigCoreAffinityReturns(int32_t value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypoint_testDTcc mht_5(mht_5_v, 271, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint_test.cc", "SetBigCoreAffinityReturns");

    big_core_affinity_result = value;
  }

  std::string storage_path_;
  FlatbufferStorage<BenchmarkEvent> storage_;
};

TEST_F(ValidatorRunnerEntryPointTest, NotEnoughArguments) {
  std::vector<std::string> args = {
      "test", "binary_name",
      "Java_org_tensorflow_lite_acceleration_validation_entrypoint",
      "model_path", storage_path_};
  std::vector<std::vector<char>> mutable_args(args.size());
  std::vector<char*> argv(args.size());
  for (int i = 0; i < mutable_args.size(); i++) {
    mutable_args[i] = {args[i].data(), args[i].data() + args[i].size()};
    mutable_args[i].push_back('\0');
    argv[i] = mutable_args[i].data();
  }
  EXPECT_EQ(1, Java_org_tensorflow_lite_acceleration_validation_entrypoint(
                   5, argv.data()));
}

TEST_F(ValidatorRunnerEntryPointTest, NoValidationRequestFound) {
  EXPECT_EQ(kMinibenchmarkSuccess, CallEntryPoint());

  std::vector<const tflite::BenchmarkEvent*> events = GetEvents();
  ASSERT_THAT(events, testing::SizeIs(1));
  const tflite::BenchmarkEvent* event = events[0];

  EXPECT_EQ(BenchmarkEventType_ERROR, event->event_type());
  EXPECT_EQ(kMinibenchmarkNoValidationRequestFound,
            event->error()->exit_code());
}

TEST_F(ValidatorRunnerEntryPointTest, CannotSetCpuAffinity) {
  SetBigCoreAffinityReturns(10);
  EXPECT_EQ(kMinibenchmarkSuccess, CallEntryPoint("invalid_cpu_affinity"));

  std::vector<const tflite::BenchmarkEvent*> events = GetEvents();
  ASSERT_THAT(events, testing::SizeIs(2));
  // The last event is the notification of NoValidationRequestFound.
  const tflite::BenchmarkEvent* event = events[0];

  EXPECT_EQ(BenchmarkEventType_RECOVERED_ERROR, event->event_type());
  EXPECT_EQ(kMinibenchmarkUnableToSetCpuAffinity, event->error()->exit_code());
  EXPECT_EQ(10, event->error()->mini_benchmark_error_code());
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
