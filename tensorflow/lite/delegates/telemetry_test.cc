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
class MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/telemetry.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace delegates {
namespace {

constexpr int32_t kDummyCode = 2;
constexpr bool kDummyGpuPrecisionLossAllowed = true;
constexpr tflite::Delegate kDummyDelegate = tflite::Delegate_GPU;
constexpr DelegateStatusSource kDummySource =
    DelegateStatusSource::TFLITE_NNAPI;

TEST(TelemetryTest, StatusConversion) {
  DelegateStatus status(kDummySource, kDummyCode);
  int64_t serialized_int = status.full_status();
  DelegateStatus deserialized_status(serialized_int);

  EXPECT_EQ(kDummyCode, deserialized_status.code());
  EXPECT_EQ(kDummySource, deserialized_status.source());
  EXPECT_EQ(serialized_int, deserialized_status.full_status());
}

// Dummy profiler to test delegate reporting.
class DelegateProfiler : public Profiler {
 public:
  DelegateProfiler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/lite/delegates/telemetry_test.cc", "DelegateProfiler");
}
  ~DelegateProfiler() override = default;

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/delegates/telemetry_test.cc", "BeginEvent");

    int event_handle = -1;
    if (event_type ==
            Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT &&
        std::string(tag) == kDelegateSettingsTag) {
      event_buffer_.emplace_back();
      event_handle = event_buffer_.size();

      // event_metadata1 is a pointer to a TfLiteDelegate.
      EXPECT_NE(event_metadata1, 0);
      auto* delegate = reinterpret_cast<TfLiteDelegate*>(event_metadata1);
      EXPECT_EQ(delegate->flags, kTfLiteDelegateFlagsNone);
      // event_metadata2 is a pointer to TFLiteSettings.
      EXPECT_NE(event_metadata2, 0);
      auto* settings = reinterpret_cast<TFLiteSettings*>(event_metadata2);
      EXPECT_EQ(settings->delegate(), kDummyDelegate);
      EXPECT_EQ(settings->gpu_settings()->is_precision_loss_allowed(),
                kDummyGpuPrecisionLossAllowed);
    } else if (event_type ==
                   Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT &&
               std::string(tag) == kDelegateStatusTag) {
      event_buffer_.emplace_back();
      event_handle = event_buffer_.size();

      EXPECT_EQ(event_metadata2, static_cast<int64_t>(kTfLiteOk));
      DelegateStatus reported_status(event_metadata1);
      EXPECT_EQ(reported_status.source(), kDummySource);
      EXPECT_EQ(reported_status.code(), kDummyCode);
    }

    EXPECT_NE(-1, event_handle);
    return event_handle;
  }

  void EndEvent(uint32_t event_handle) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc mht_2(mht_2_v, 265, "", "./tensorflow/lite/delegates/telemetry_test.cc", "EndEvent");

    EXPECT_EQ(event_handle, event_buffer_.size());
  }

  int NumRecordedEvents() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetry_testDTcc mht_3(mht_3_v, 272, "", "./tensorflow/lite/delegates/telemetry_test.cc", "NumRecordedEvents");
 return event_buffer_.size(); }

 private:
  std::vector<profiling::ProfileEvent> event_buffer_;
};

TEST(TelemetryTest, DelegateStatusReport) {
  DelegateProfiler profiler;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  TfLiteContext context;
  context.profiler = &profiler;
  DelegateStatus status(kDummySource, kDummyCode);

  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 2);
}

TEST(TelemetryTest, DelegateSettingsReport) {
  DelegateProfiler profiler;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  TfLiteContext context;
  context.profiler = &profiler;

  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffers::Offset<tflite::GPUSettings> gpu_settings =
      tflite::CreateGPUSettings(
          flatbuffer_builder,
          /**is_precision_loss_allowed**/ kDummyGpuPrecisionLossAllowed);
  auto* tflite_settings_ptr = flatbuffers::GetTemporaryPointer(
      flatbuffer_builder,
      CreateTFLiteSettings(flatbuffer_builder, kDummyDelegate,
                           /*nnapi_settings=*/0,
                           /*gpu_settings=*/gpu_settings));

  EXPECT_EQ(ReportDelegateSettings(&context, &delegate, *tflite_settings_ptr),
            kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 1);

  // Also report status to simulate typical use-case.
  DelegateStatus status(kDummySource, kDummyCode);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(ReportDelegateStatus(&context, &delegate, status), kTfLiteOk);
  EXPECT_EQ(profiler.NumRecordedEvents(), 3);
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
