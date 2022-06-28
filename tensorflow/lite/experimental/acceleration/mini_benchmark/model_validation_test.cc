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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc() {
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
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"

extern const unsigned char TENSORFLOW_ACCELERATION_MODEL_DATA_VARIABLE[];
extern const int TENSORFLOW_ACCELERATION_MODEL_LENGTH_VARIABLE;

namespace tflite {
namespace acceleration {
namespace {

class LocalizerValidationRegressionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/model_validation_test.cc", "SetUpTestSuite");

    int error = -1;
#if defined(__ANDROID__)
    AndroidInfo android_info;
    auto status = RequestAndroidInfo(&android_info);
    ASSERT_TRUE(status.ok());
    if (android_info.is_emulator) {
      std::cerr << "Running on an emulator, skipping processor affinity\n";
    } else {
      std::cerr << "Running on hardware, setting processor affinity\n";
      BigLittleAffinity affinity = GetAffinity();
      cpu_set_t set;
      CPU_ZERO(&set);
      for (int i = 0; i < 16; i++) {
        if (affinity.big_core_affinity & (0x1 << i)) {
          CPU_SET(i, &set);
        }
      }
      error = sched_setaffinity(getpid(), sizeof(set), &set);
      if (error == -1) {
        perror("sched_setaffinity failed");
      }
    }
#endif
    std::string dir = GetTestTmpDir();
    error = mkdir(dir.c_str(), 0777);
    if (error == -1) {
      if (errno != EEXIST) {
        perror("mkdir failed");
        ASSERT_TRUE(false);
      }
    }

    std::string path = ModelPath();
    (void)unlink(path.c_str());
    std::string contents(reinterpret_cast<const char*>(
                             TENSORFLOW_ACCELERATION_MODEL_DATA_VARIABLE),
                         TENSORFLOW_ACCELERATION_MODEL_LENGTH_VARIABLE);
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f << contents;
    f.close();
    ASSERT_EQ(chmod(path.c_str(), 0500), 0);
  }
  static std::string ModelPath() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc mht_1(mht_1_v, 257, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/model_validation_test.cc", "ModelPath");
 return GetTestTmpDir() + "/model.tflite"; }
  static std::string GetTestTmpDir() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc mht_2(mht_2_v, 261, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/model_validation_test.cc", "GetTestTmpDir");

    const char* from_env = getenv("TEST_TMPDIR");
    if (from_env) {
      return from_env;
    }
#ifdef __ANDROID__
    return "/data/local/tmp";
#else
    return "/tmp";
#endif
  }
  void CheckValidation(const std::string& accelerator_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("accelerator_name: \"" + accelerator_name + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmodel_validation_testDTcc mht_3(mht_3_v, 276, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/model_validation_test.cc", "CheckValidation");

    std::string path = ModelPath();
    const ComputeSettings* settings =
        flatbuffers::GetRoot<ComputeSettings>(fbb_.GetBufferPointer());
    int fd = open(path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    struct stat stat_buf = {0};
    ASSERT_EQ(fstat(fd, &stat_buf), 0);
    auto validator =
        std::make_unique<Validator>(fd, 0, stat_buf.st_size, settings);
    close(fd);

    Validator::Results results;
    EXPECT_EQ(validator->RunValidation(&results), kMinibenchmarkSuccess);
    EXPECT_TRUE(results.ok);
    EXPECT_EQ(results.delegate_error, 0);

    for (const auto& metric : results.metrics) {
      int test_case = 0;
      std::cerr << "Metric " << metric.first;
      for (float v : metric.second) {
        std::cerr << " " << v;
        RecordProperty("[" + std::to_string(test_case++) + "] " +
                           accelerator_name + " " + metric.first,
                       std::to_string(v));
      }
      std::cerr << "\n";
    }
    std::cerr << "Compilation time us " << results.compilation_time_us
              << std::endl;
    RecordProperty(accelerator_name + " Compilation time us",
                   results.compilation_time_us);
    std::cerr << "Execution time us";
    int test_case = 0;
    for (int64_t t : results.execution_time_us) {
      std::cerr << " " << t;
      RecordProperty("[" + std::to_string(test_case++) + "] " +
                         accelerator_name + " Execution time us",
                     t);
    }
    std::cerr << std::endl;
  }
  flatbuffers::FlatBufferBuilder fbb_;
};

TEST_F(LocalizerValidationRegressionTest, Cpu) {
  fbb_.Finish(CreateComputeSettings(fbb_, ExecutionPreference_ANY,
                                    CreateTFLiteSettings(fbb_)));
  CheckValidation("CPU");
}

TEST_F(LocalizerValidationRegressionTest, Nnapi) {
  fbb_.Finish(
      CreateComputeSettings(fbb_, ExecutionPreference_ANY,
                            CreateTFLiteSettings(fbb_, Delegate_NNAPI)));
  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());
  if (android_info.android_sdk_version >= "28") {
    CheckValidation("NNAPI");
  }
}

TEST_F(LocalizerValidationRegressionTest, Gpu) {
  AndroidInfo android_info;
  auto status = RequestAndroidInfo(&android_info);
  ASSERT_TRUE(status.ok());
  if (android_info.is_emulator) {
    std::cerr << "Skipping GPU on emulator\n";
    return;
  }
  fbb_.Finish(CreateComputeSettings(fbb_, ExecutionPreference_ANY,
                                    CreateTFLiteSettings(fbb_, Delegate_GPU)));
#ifdef __ANDROID__
  CheckValidation("GPU");
#endif  // __ANDROID__
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
