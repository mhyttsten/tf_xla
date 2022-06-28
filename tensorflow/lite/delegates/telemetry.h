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
#ifndef TENSORFLOW_LITE_DELEGATES_STATUS_H_
#define TENSORFLOW_LITE_DELEGATES_STATUS_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh() {
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


#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

// This file implements utilities for delegate telemetry. These enable
// representation and reporting of hardware-specific configurations, status
// codes, etc.
// These APIs are for internal use *only*, and should be modified with care to
// avoid incompatibilities between delegates & runtime.
// WARNING: This is an experimental feature that is subject to change.
namespace tflite {
namespace delegates {

// Used to identify specific events for tflite::Profiler.
constexpr char kDelegateSettingsTag[] = "delegate_settings";
constexpr char kDelegateStatusTag[] = "delegate_status";

// Defines the delegate or hardware-specific 'namespace' that a status code
// belongs to. For example, GPU delegate errors might be belong to TFLITE_GPU,
// while OpenCL-specific ones might be TFLITE_GPU_CL.
enum class DelegateStatusSource {
  NONE = 0,
  TFLITE_GPU = 1,
  TFLITE_NNAPI = 2,
  TFLITE_HEXAGON = 3,
  TFLITE_XNNPACK = 4,
  TFLITE_COREML = 5,
  MAX_NUM_SOURCES = std::numeric_limits<int32_t>::max(),
};

// DelegateStatus defines a namespaced status with a combination of
// DelegateStatusSource & the corresponding fine-grained 32-bit code. Used to
// convert to/from a 64-bit representation as follows:
//
// delegates::DelegateStatus status(
//      delegates::DelegateStatusSource::TFLITE_NNAPI,
//      ANEURALNETWORKS_OP_FAILED);
// int64_t code = status.full_status();
//
// auto parsed_status = delegates::DelegateStatus(code);
class DelegateStatus {
 public:
  DelegateStatus() : DelegateStatus(DelegateStatusSource::NONE, 0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_0(mht_0_v, 231, "", "./tensorflow/lite/delegates/telemetry.h", "DelegateStatus");
}
  explicit DelegateStatus(int32_t code)
      : DelegateStatus(DelegateStatusSource::NONE, code) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_1(mht_1_v, 236, "", "./tensorflow/lite/delegates/telemetry.h", "DelegateStatus");
}
  explicit DelegateStatus(int64_t full_status)
      : DelegateStatus(
            static_cast<DelegateStatusSource>(
                full_status >> 32 &
                static_cast<int32_t>(DelegateStatusSource::MAX_NUM_SOURCES)),
            static_cast<int32_t>(full_status &
                                 std::numeric_limits<int32_t>::max())) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_2(mht_2_v, 246, "", "./tensorflow/lite/delegates/telemetry.h", "DelegateStatus");
}
  DelegateStatus(DelegateStatusSource source, int32_t code)
      : source_(static_cast<int32_t>(source)), code_(code) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_3(mht_3_v, 251, "", "./tensorflow/lite/delegates/telemetry.h", "DelegateStatus");
}

  // Return the detailed full status encoded as a int64_t value.
  int64_t full_status() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_4(mht_4_v, 257, "", "./tensorflow/lite/delegates/telemetry.h", "full_status");

    return static_cast<int64_t>(source_) << 32 | code_;
  }

  DelegateStatusSource source() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/telemetry.h", "source");

    return static_cast<DelegateStatusSource>(source_);
  }

  int32_t code() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPStelemetryDTh mht_6(mht_6_v, 271, "", "./tensorflow/lite/delegates/telemetry.h", "code");
 return code_; }

 private:
  // value of a DelegateStatusSource, like DelegateStatusSource::TFLITE_GPU
  int32_t source_;
  // value of a status code, like kTfLiteOk.
  int32_t code_;
};

// Used by delegates to report their configuration/settings to TFLite.
// Calling this method adds a new GENERAL_RUNTIME_INSTRUMENTATION_EVENT to
// the runtime Profiler.
TfLiteStatus ReportDelegateSettings(TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    const TFLiteSettings& settings);

// Used by delegates to report their status to the TFLite runtime.
// Calling this method adds a new GENERAL_RUNTIME_INSTRUMENTATION_EVENT to
// the runtime Profiler.
TfLiteStatus ReportDelegateStatus(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  const DelegateStatus& status);

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_STATUS_H_
