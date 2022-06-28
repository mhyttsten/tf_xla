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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.h"

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers

namespace tflite {
namespace acceleration {

namespace {
class NoopMiniBenchmark : public MiniBenchmark {
 public:
  ComputeSettingsT GetBestAcceleration() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "GetBestAcceleration");
 return ComputeSettingsT(); }
  void TriggerMiniBenchmark() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_1(mht_1_v, 202, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "TriggerMiniBenchmark");
}
  void SetEventTimeoutForTesting(int64_t) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_2(mht_2_v, 206, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "SetEventTimeoutForTesting");
}
  std::vector<MiniBenchmarkEventT> MarkAndGetEventsToLog() override {
    return {};
  }
  // We return -1 as this no-op instance doesn't have the overall
  // mini-benchmark-related setup properly initialized.
  int NumRemainingAccelerationTests() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_3(mht_3_v, 215, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "NumRemainingAccelerationTests");
 return -1; }
};
}  // namespace

std::unique_ptr<MiniBenchmark> CreateMiniBenchmark(
    const MinibenchmarkSettings& settings, const std::string& model_namespace,
    const std::string& model_id) {
  absl::StatusOr<std::unique_ptr<MiniBenchmark>> s_or_mb =
      MinibenchmarkImplementationRegistry::CreateByName(
          "Impl", settings, model_namespace, model_id);
  if (!s_or_mb.ok()) {
    return std::unique_ptr<MiniBenchmark>(new NoopMiniBenchmark());
  } else {
    return std::move(*s_or_mb);
  }
}

void MinibenchmarkImplementationRegistry::RegisterImpl(
    const std::string& name, CreatorFunction creator_function) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_4(mht_4_v, 237, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "MinibenchmarkImplementationRegistry::RegisterImpl");

  absl::MutexLock lock(&mutex_);
  factories_[name] = creator_function;
}

std::unique_ptr<MiniBenchmark> MinibenchmarkImplementationRegistry::CreateImpl(
    const std::string& name, const MinibenchmarkSettings& settings,
    const std::string& model_namespace, const std::string& model_id) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   mht_5_v.push_back("model_namespace: \"" + model_namespace + "\"");
   mht_5_v.push_back("model_id: \"" + model_id + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_5(mht_5_v, 250, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "MinibenchmarkImplementationRegistry::CreateImpl");

  absl::MutexLock lock(&mutex_);
  auto it = factories_.find(name);
  return (it != factories_.end())
             ? it->second(settings, model_namespace, model_id)
             : nullptr;
}

MinibenchmarkImplementationRegistry*
MinibenchmarkImplementationRegistry::GetSingleton() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_6(mht_6_v, 262, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "MinibenchmarkImplementationRegistry::GetSingleton");

  static auto* instance = new MinibenchmarkImplementationRegistry();
  return instance;
}

std::unique_ptr<MiniBenchmark>
MinibenchmarkImplementationRegistry::CreateByName(
    const std::string& name, const MinibenchmarkSettings& settings,
    const std::string& model_namespace, const std::string& model_id) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("model_namespace: \"" + model_namespace + "\"");
   mht_7_v.push_back("model_id: \"" + model_id + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_7(mht_7_v, 276, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "MinibenchmarkImplementationRegistry::CreateByName");

  auto* const instance = MinibenchmarkImplementationRegistry::GetSingleton();
  return instance->CreateImpl(name, settings, model_namespace, model_id);
}

MinibenchmarkImplementationRegistry::Register::Register(
    const std::string& name, CreatorFunction creator_function) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmarkDTcc mht_8(mht_8_v, 286, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.cc", "MinibenchmarkImplementationRegistry::Register::Register");

  auto* const instance = MinibenchmarkImplementationRegistry::GetSingleton();
  instance->RegisterImpl(name, creator_function);
}

}  // namespace acceleration
}  // namespace tflite
