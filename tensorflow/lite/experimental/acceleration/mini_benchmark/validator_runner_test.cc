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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

#ifdef __ANDROID__
#include <dlfcn.h>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_entrypoint.h"
#endif  // __ANDROID__

namespace tflite {
namespace acceleration {
namespace {

std::vector<const TFLiteSettings*> BuildBenchmarkSettings(
    const AndroidInfo& android_info, flatbuffers::FlatBufferBuilder& fbb_cpu,
    flatbuffers::FlatBufferBuilder& fbb_nnapi,
    flatbuffers::FlatBufferBuilder& fbb_gpu,
    bool ignore_android_version = false) {
  std::vector<const TFLiteSettings*> settings;
  fbb_cpu.Finish(CreateTFLiteSettings(fbb_cpu, Delegate_NONE,
                                      CreateNNAPISettings(fbb_cpu)));
  settings.push_back(
      flatbuffers::GetRoot<TFLiteSettings>(fbb_cpu.GetBufferPointer()));
  if (ignore_android_version || android_info.android_sdk_version >= "28") {
    fbb_nnapi.Finish(CreateTFLiteSettings(fbb_nnapi, Delegate_NNAPI,
                                          CreateNNAPISettings(fbb_nnapi)));
    settings.push_back(
        flatbuffers::GetRoot<TFLiteSettings>(fbb_nnapi.GetBufferPointer()));
  }

#ifdef __ANDROID__
  fbb_gpu.Finish(CreateTFLiteSettings(fbb_gpu, Delegate_GPU));
  settings.push_back(
      flatbuffers::GetRoot<TFLiteSettings>(fbb_gpu.GetBufferPointer()));
#endif  // __ANDROID__

  return settings;
}

std::string GetTargetDeviceName(const BenchmarkEvent* event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc mht_0(mht_0_v, 240, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_test.cc", "GetTargetDeviceName");

  if (event->tflite_settings()->delegate() == Delegate_GPU) {
    return "GPU";
  } else if (event->tflite_settings()->delegate() == Delegate_NNAPI) {
    return "NNAPI";
  }
  return "CPU";
}

class ValidatorRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc mht_1(mht_1_v, 254, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_test.cc", "SetUp");

    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (!should_perform_test_) {
      return;
    }

    model_path_ = helper.DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!model_path_.empty());

#ifdef __ANDROID__
    // We extract the test files here as that's the only way to get the right
    // architecture when building tests for multiple architectures.
    std::string entry_point_file = MiniBenchmarkTestHelper::DumpToTempFile(
        "libvalidator_runner_entrypoint.so",
        g_tflite_acceleration_embedded_validator_runner_entrypoint,
        g_tflite_acceleration_embedded_validator_runner_entrypoint_len);
    ASSERT_TRUE(!entry_point_file.empty());

    void* module =
        dlopen(entry_point_file.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    EXPECT_TRUE(module) << dlerror();
#endif  // __ANDROID__
  }

  void CheckConfigurations(bool use_path = true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_testDTcc mht_2(mht_2_v, 286, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_test.cc", "CheckConfigurations");

    if (!should_perform_test_) {
      std::cerr << "Skipping test";
      return;
    }
    AndroidInfo android_info;
    auto status = RequestAndroidInfo(&android_info);
    ASSERT_TRUE(status.ok());

    std::unique_ptr<ValidatorRunner> validator1, validator2;
    std::string storage_path = ::testing::TempDir() + "/storage_path.fb";
    (void)unlink(storage_path.c_str());
    if (use_path) {
      validator1 = std::make_unique<ValidatorRunner>(model_path_, storage_path,
                                                     ::testing::TempDir());
      validator2 = std::make_unique<ValidatorRunner>(model_path_, storage_path,
                                                     ::testing::TempDir());
    } else {
      int fd = open(model_path_.c_str(), O_RDONLY);
      ASSERT_GE(fd, 0);
      struct stat stat_buf = {0};
      ASSERT_EQ(fstat(fd, &stat_buf), 0);
      validator1 = std::make_unique<ValidatorRunner>(
          fd, 0, stat_buf.st_size, storage_path, ::testing::TempDir());
      validator2 = std::make_unique<ValidatorRunner>(
          fd, 0, stat_buf.st_size, storage_path, ::testing::TempDir());
    }
    ASSERT_EQ(validator1->Init(), kMinibenchmarkSuccess);
    ASSERT_EQ(validator2->Init(), kMinibenchmarkSuccess);

    std::vector<const BenchmarkEvent*> events =
        validator1->GetAndFlushEventsToLog();
    ASSERT_TRUE(events.empty());

    flatbuffers::FlatBufferBuilder fbb_cpu, fbb_nnapi, fbb_gpu;
    std::vector<const TFLiteSettings*> settings =
        BuildBenchmarkSettings(android_info, fbb_cpu, fbb_nnapi, fbb_gpu);

    ASSERT_EQ(validator1->TriggerMissingValidation(settings), settings.size());

    int event_count = 0;
    while (event_count < settings.size()) {
      events = validator1->GetAndFlushEventsToLog();
      event_count += events.size();
      for (const BenchmarkEvent* event : events) {
        std::string delegate_name = GetTargetDeviceName(event);
        if (event->event_type() == BenchmarkEventType_END) {
          if (event->result()->ok()) {
            std::cout << "Validation passed on " << delegate_name << std::endl;
          } else {
            std::cout << "Validation did not pass on " << delegate_name
                      << std::endl;
          }
        } else if (event->event_type() == BenchmarkEventType_ERROR) {
          std::cout << "Failed to run validation on " << delegate_name
                    << std::endl;
        }
      }
#ifndef _WIN32
      sleep(1);
#endif  // !_WIN32
    }

    EXPECT_EQ(validator2->TriggerMissingValidation(settings), 0);
  }

  bool should_perform_test_ = true;
  std::string model_path_;
};

TEST_F(ValidatorRunnerTest, AllConfigurationsWithFilePath) {
  CheckConfigurations(true);
}

TEST_F(ValidatorRunnerTest, AllConfigurationsWithFd) {
  CheckConfigurations(false);
}

// #ifdef __ANDROID__
using ::tflite::nnapi::NnApiSupportLibrary;

std::unique_ptr<const NnApiSupportLibrary> LoadNnApiSupportLibrary() {
  MiniBenchmarkTestHelper helper;
  std::string nnapi_sl_path = helper.DumpToTempFile(
      "libnnapi_fake.so", g_nnapi_sl_fake_impl, g_nnapi_sl_fake_impl_len);

  std::unique_ptr<const NnApiSupportLibrary> nnapi_sl =
      ::tflite::nnapi::loadNnApiSupportLibrary(nnapi_sl_path);

  return nnapi_sl;
}

TEST_F(ValidatorRunnerTest, ShouldUseNnApiSl) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());

  InitNnApiSlInvocationStatus();

  std::string storage_path = ::testing::TempDir() + "/storage_path.fb";
  (void)unlink(storage_path.c_str());

  std::unique_ptr<const NnApiSupportLibrary> nnapi_sl =
      LoadNnApiSupportLibrary();
  ASSERT_THAT(nnapi_sl.get(), ::testing::NotNull());
  ValidatorRunner validator(model_path_, storage_path, ::testing::TempDir(),
                            nnapi_sl->getFL5());

  ASSERT_EQ(validator.Init(), kMinibenchmarkSuccess);

  std::vector<const BenchmarkEvent*> events =
      validator.GetAndFlushEventsToLog();
  ASSERT_TRUE(events.empty());

  flatbuffers::FlatBufferBuilder fbb_cpu, fbb_nnapi, fbb_gpu;
  std::vector<const TFLiteSettings*> settings =
      BuildBenchmarkSettings(android_info, fbb_cpu, fbb_nnapi, fbb_gpu,
                             /*ignore_android_version=*/true);
  ASSERT_EQ(validator.TriggerMissingValidation(settings), settings.size());

  // Waiting for benchmark to complete.
  int event_count = 0;
  while (event_count < settings.size()) {
    events = validator.GetAndFlushEventsToLog();
    event_count += events.size();
  }
  EXPECT_TRUE(WasNnApiSlInvoked());
}

TEST_F(ValidatorRunnerTest, ShouldFailIfItCannotFindNnApiSlPath) {
  if (!should_perform_test_) {
    std::cerr << "Skipping test";
    return;
  }

  std::string storage_path = ::testing::TempDir() + "/storage_path.fb";
  (void)unlink(storage_path.c_str());

  // Building an NNAPI SL structure with invalid handle.
  NnApiSLDriverImplFL5 wrong_handle_nnapi_sl{};

  ValidatorRunner validator(model_path_, storage_path, ::testing::TempDir(),
                            &wrong_handle_nnapi_sl);

  ASSERT_EQ(validator.Init(), kMiniBenchmarkCannotLoadSupportLibrary);
}
// #endif  // ifdef __ANDROID__

}  // namespace
}  // namespace acceleration
}  // namespace tflite
