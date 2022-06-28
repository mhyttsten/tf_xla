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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storage_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storage_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storage_testDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"

#include <thread>  // NOLINT - only production use is on Android, where std::thread is allowed

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {
namespace {

std::string GetTemporaryDirectory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storage_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage_test.cc", "GetTemporaryDirectory");

#ifdef __ANDROID__
  return "/data/local/tmp";
#else
  if (getenv("TEST_TMPDIR")) {
    return getenv("TEST_TMPDIR");
  }
  return getenv("TEMP");
#endif
}

std::string GetStoragePath() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storage_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage_test.cc", "GetStoragePath");

  std::string path = GetTemporaryDirectory() + "/storage.fb";
  unlink(path.c_str());
  return path;
}

TEST(FlatbufferStorageTest, AppendAndReadOneItem) {
  std::string path = GetStoragePath();
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<BenchmarkEvent> o =
      CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);

  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), 0);

  EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);

  storage = FlatbufferStorage<BenchmarkEvent>(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
}

TEST(FlatbufferStorageTest, AppendAndReadThreeItems) {
  std::string path = GetStoragePath();
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), 0);

  for (auto event : {BenchmarkEventType_START, BenchmarkEventType_ERROR,
                     BenchmarkEventType_END}) {
    flatbuffers::FlatBufferBuilder fbb;
    flatbuffers::Offset<BenchmarkEvent> object =
        CreateBenchmarkEvent(fbb, 0, event);
    EXPECT_EQ(storage.Append(&fbb, object), kMinibenchmarkSuccess);
  }

  ASSERT_EQ(storage.Count(), 3);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
  EXPECT_EQ(storage.Get(1)->event_type(), BenchmarkEventType_ERROR);
  EXPECT_EQ(storage.Get(2)->event_type(), BenchmarkEventType_END);

  storage = FlatbufferStorage<BenchmarkEvent>(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 3);
  EXPECT_EQ(storage.Get(0)->event_type(), BenchmarkEventType_START);
  EXPECT_EQ(storage.Get(1)->event_type(), BenchmarkEventType_ERROR);
  EXPECT_EQ(storage.Get(2)->event_type(), BenchmarkEventType_END);
}

TEST(FlatbufferStorageTest, PathDoesntExist) {
  std::string path = GetTemporaryDirectory() + "/nosuchdirectory/storage.pb";
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkCantCreateStorageFile);
}

#ifndef __ANDROID__
// chmod(0444) doesn't block writing on Android.
TEST(FlatbufferStorageTest, WriteFailureResetsStorage) {
  std::string path = GetStoragePath();
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<BenchmarkEvent> o =
      CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);

  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
  ASSERT_EQ(storage.Count(), 1);

  chmod(path.c_str(), 0444);
  EXPECT_EQ(storage.Append(&fbb, o),
            kMinibenchmarkFailedToOpenStorageFileForWriting);
  ASSERT_EQ(storage.Count(), 0);
}
#endif  // !__ANDROID__

TEST(FlatbufferStorageTest, Locking) {
  std::string path = GetStoragePath();

  std::vector<std::thread> threads;
  const int kNumThreads = 4;
  const int kIterations = 10;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(std::thread([path]() {
      for (int j = 0; j < kIterations; j++) {
        FlatbufferStorage<BenchmarkEvent> storage(path);
        flatbuffers::FlatBufferBuilder fbb;
        flatbuffers::Offset<BenchmarkEvent> o =
            CreateBenchmarkEvent(fbb, 0, BenchmarkEventType_START);
        EXPECT_EQ(storage.Append(&fbb, o), kMinibenchmarkSuccess);
      }
    }));
  }
  std::for_each(threads.begin(), threads.end(),
                [](std::thread& t) { t.join(); });
  FlatbufferStorage<BenchmarkEvent> storage(path);
  EXPECT_EQ(storage.Read(), kMinibenchmarkSuccess);
  EXPECT_EQ(storage.Count(), kNumThreads * kIterations);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
